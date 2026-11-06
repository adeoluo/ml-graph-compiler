"""Compile generated C into a shared library with the system C compiler.

The compiler is ``$MGC_CC``, else ``$CC``, else the first of clang, cc and gcc on the PATH.
OpenMP is used when a probe program builds and links: ``-fopenmp`` with GCC and Linux clang,
and ``-Xpreprocessor -fopenmp`` plus Homebrew's libomp with Apple clang. Libraries are cached
under a key that hashes the sources and the flags, so compiling an unchanged model is free.
"""

from __future__ import annotations

import functools
import hashlib
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from ml_graph_compiler.errors import CodegenError

RUNTIME_DIR = Path(__file__).parent / "runtime"
RUNTIME_FILES = ("mgc_runtime.h", "mgc_kernels.c")
LIB_SUFFIX = ".dylib" if sys.platform == "darwin" else ".so"


@dataclass(frozen=True)
class BuildOptions:
    """C compiler settings. ``openmp=None`` means use OpenMP when available."""

    opt_level: str = "-O3"
    native: bool = True
    openmp: bool | None = None
    neon: bool = True
    extra_flags: tuple[str, ...] = field(default_factory=tuple)


def find_compiler() -> str:
    for candidate in (os.environ.get("MGC_CC"), os.environ.get("CC"), "clang", "cc", "gcc"):
        if candidate and shutil.which(candidate):
            return candidate
    raise CodegenError("no C compiler found (set MGC_CC or CC, or install clang)")


def _homebrew_libomp() -> Path | None:
    for prefix in (
        os.environ.get("MGC_LIBOMP"),
        "/opt/homebrew/opt/libomp",
        "/usr/local/opt/libomp",
    ):
        if prefix and (Path(prefix) / "include" / "omp.h").exists():
            return Path(prefix)
    return None


def _openmp_candidates(cc: str) -> list[tuple[list[str], list[str]]]:
    candidates: list[tuple[list[str], list[str]]] = []
    if sys.platform == "darwin":
        libomp = _homebrew_libomp()
        if libomp is not None:
            candidates.append(
                (
                    ["-Xpreprocessor", "-fopenmp", f"-I{libomp / 'include'}"],
                    [f"-L{libomp / 'lib'}", "-lomp", f"-Wl,-rpath,{libomp / 'lib'}"],
                )
            )
    candidates.append((["-fopenmp"], ["-fopenmp"]))
    return candidates


@functools.cache
def openmp_flags(cc: str, workdir: str) -> tuple[tuple[str, ...], tuple[str, ...]] | None:
    """(compile flags, link flags) that build an OpenMP program with ``cc``, or None."""
    probe = Path(workdir) / "mgc_omp_probe.c"
    probe.parent.mkdir(parents=True, exist_ok=True)
    probe.write_text(
        "#include <omp.h>\nint f(void){int s=0;\n#pragma omp parallel reduction(+:s)\ns+=1;\n"
        "return s+omp_get_max_threads();}\n"
    )
    out = probe.with_suffix(LIB_SUFFIX)
    for cflags, ldflags in _openmp_candidates(cc):
        cmd = [cc, "-shared", "-fPIC", *cflags, str(probe), "-o", str(out), *ldflags]
        if subprocess.run(cmd, capture_output=True, check=False).returncode == 0:
            return tuple(cflags), tuple(ldflags)
    return None


def arch_flags(cc: str) -> list[str]:
    machine = platform.machine().lower()
    if machine in ("arm64", "aarch64"):
        return ["-mcpu=native"] if _accepts(cc, "-mcpu=native") else []
    return ["-march=native"] if _accepts(cc, "-march=native") else []


@functools.cache
def _accepts(cc: str, flag: str) -> bool:
    result = subprocess.run(
        [cc, flag, "-x", "c", "-c", "-o", os.devnull, "-"],
        input=b"int x;\n",
        capture_output=True,
        check=False,
    )
    return result.returncode == 0


def default_cache_dir() -> Path:
    base = os.environ.get("MGC_CACHE_DIR")
    if base:
        return Path(base)
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "ml-graph-compiler"


@dataclass
class BuildResult:
    library: Path
    directory: Path
    command: list[str]
    cached: bool
    openmp: bool


def _run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise CodegenError(f"C compilation failed:\n{' '.join(cmd)}\n{proc.stderr[-4000:]}")


def build_library(
    source: str,
    options: BuildOptions | None = None,
    cache_dir: Path | None = None,
    name: str = "model",
) -> BuildResult:
    """Build ``source`` with the runtime kernels into a shared library (both steps cached).

    The kernels are compiled once per compiler and flag set into an object file; each model only
    compiles its own generated source and links.
    """
    options = options or BuildOptions()
    cc = find_compiler()
    cache = Path(cache_dir) if cache_dir is not None else default_cache_dir()
    omp = None
    if options.openmp is not False:
        omp = openmp_flags(cc, str(cache / "probe"))
        if options.openmp and omp is None:
            raise CodegenError(f"OpenMP was requested but {cc} cannot build OpenMP programs")
    cflags = [
        options.opt_level,
        "-std=c11",
        "-fPIC",
        "-ffp-contract=fast",
        "-fno-math-errno",
        "-Wall",
        "-Wno-unused-variable",
        "-Wno-unused-function",
    ]
    if options.native:
        cflags += arch_flags(cc)
    if not options.neon:
        cflags.append("-DMGC_NO_NEON")
    cflags += list(options.extra_flags)
    ldflags = ["-lm"]
    if omp is not None:
        cflags += list(omp[0])
        ldflags = list(omp[1]) + ldflags
    runtime = {f: (RUNTIME_DIR / f).read_text() for f in RUNTIME_FILES}

    def key(*parts: str) -> str:
        digest = hashlib.sha256()
        for part in parts:
            digest.update(part.encode())
            digest.update(b"\0")
        return digest.hexdigest()[:20]

    kernels_dir = cache / f"kernels-{key(*runtime.values(), cc, *cflags)}"
    kernels_obj = kernels_dir / "mgc_kernels.o"
    if not kernels_obj.exists():
        kernels_dir.mkdir(parents=True, exist_ok=True)
        for fname, text in runtime.items():
            (kernels_dir / fname).write_text(text)
        tmp = kernels_obj.with_suffix(f".tmp{os.getpid()}.o")
        _run([cc, *cflags, "-c", str(kernels_dir / "mgc_kernels.c"), "-o", str(tmp)])
        tmp.replace(kernels_obj)

    directory = cache / f"{name}-{key(source, *runtime.values(), cc, *cflags, *ldflags)}"
    library = directory / f"lib{name}{LIB_SUFFIX}"
    command = [
        cc,
        *cflags,
        "-shared",
        "-I",
        str(kernels_dir),
        str(directory / f"{name}.c"),
        str(kernels_obj),
        "-o",
        str(library),
        *ldflags,
    ]
    if library.exists():
        return BuildResult(library, directory, command, True, omp is not None)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}.c").write_text(source)
    for fname, text in runtime.items():
        (directory / fname).write_text(text)
    tmp = library.with_suffix(f".tmp{os.getpid()}{LIB_SUFFIX}")
    _run([*command[: -len(ldflags) - 2], "-o", str(tmp), *ldflags])
    tmp.replace(library)
    return BuildResult(library, directory, command, False, omp is not None)
