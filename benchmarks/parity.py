"""Numerical parity of the compiled models against onnxruntime on real inputs.

ResNet-50 and MobileNetV2 run on every image of the Imagenette validation split (3,925 ImageNet
validation-style photos of 10 ImageNet classes, 320 px version), preprocessed the torchvision way.
BERT-base and GPT-2 run on random token sequences of length 128 (seeded). For each input the C
backend at -O3 and onnxruntime (all graph optimizations) run the same tensor; the script records
the max absolute error, the error relative to the output's max magnitude, top-1 agreement, and for
the vision models the top-1 accuracy of both engines against the Imagenette labels. A sample of
inputs also runs through the numpy reference interpreter at -O0 (unoptimized graph) as a third
opinion.

    uv run --group bench python benchmarks/parity.py [--models resnet50 ...] [--limit N]
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from common import DATA, exit_now, model_path, peak_rss_mb, write_result

from ml_graph_compiler.codegen import compile_model
from ml_graph_compiler.interpreter import run_graph
from ml_graph_compiler.onnx_import import from_onnx

IMAGENETTE = DATA / "imagenette" / "imagenette2-320" / "val"
# Imagenette folder (WordNet id) -> ImageNet-1k class index.
WNID_TO_INDEX = {
    "n01440764": 0,
    "n02102040": 217,
    "n02979186": 482,
    "n03000684": 491,
    "n03028079": 497,
    "n03394916": 566,
    "n03417042": 569,
    "n03425413": 571,
    "n03445777": 574,
    "n03888257": 701,
}
MEAN = np.array([0.485, 0.456, 0.406], np.float32)[:, None, None]
STD = np.array([0.229, 0.224, 0.225], np.float32)[:, None, None]
SEQ = 128
VOCAB = {"bert-base": 30522, "gpt2": 50257}


def preprocess(path: Path) -> np.ndarray:
    """Resize the short side to 256 (bilinear), center crop 224, scale to [0, 1], normalize."""
    from PIL import Image

    img = Image.open(path).convert("RGB")
    w, h = img.size
    s = 256 / min(w, h)
    img = img.resize((round(w * s), round(h * s)), Image.BILINEAR)
    w, h = img.size
    left, top = (w - 224) // 2, (h - 224) // 2
    img = img.crop((left, top, left + 224, top + 224))
    x = np.asarray(img, np.float32).transpose(2, 0, 1) / 255.0
    return ((x - MEAN) / STD)[None].astype(np.float32)


def imagenette(limit: int | None) -> Iterator[tuple[str, int, np.ndarray]]:
    files = sorted(IMAGENETTE.glob("*/*.JPEG"))
    if limit:
        # Evenly spaced so that a limited run still covers every class.
        files = [files[i] for i in np.linspace(0, len(files) - 1, limit).astype(int)]
    for f in files:
        yield f.name, WNID_TO_INDEX[f.parent.name], preprocess(f)


def token_inputs(name: str, n: int) -> Iterator[tuple[str, int, np.ndarray]]:
    for seed in range(n):
        rng = np.random.default_rng(seed)
        yield f"seed{seed}", -1, rng.integers(0, VOCAB[name], (1, SEQ), dtype=np.int64)


def ort_session(path: Path, threads: int) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def evaluate(name: str, limit: int | None, n_text: int, interp_every: int) -> dict[str, Any]:
    path = model_path(name)
    vision = name in ("resnet50", "mobilenetv2")
    start = time.perf_counter()
    compiled = compile_model(str(path), opt="O3")
    compile_seconds = time.perf_counter() - start
    sess = ort_session(path, 8)
    in_name = sess.get_inputs()[0].name
    out_names = [o.name for o in sess.get_outputs()]
    reference_graph = from_onnx(str(path)) if interp_every else None

    stats: dict[str, dict[str, Any]] = {
        o: {"max_abs": 0.0, "max_rel": 0.0, "sum_abs": 0.0, "count": 0} for o in out_names
    }
    agree = total_positions = correct_mgc = correct_ort = n = 0
    interp = {"inputs": 0, "max_abs_vs_ort": 0.0, "top1_agree": 0, "positions": 0}
    inputs = imagenette(limit) if vision else token_inputs(name, n_text)
    for i, (_, label, x) in enumerate(inputs):
        feeds = {in_name: x}
        ref = dict(zip(out_names, sess.run(None, feeds), strict=True))
        ours = compiled.run(feeds)
        for o in out_names:
            diff = np.abs(ours[o].astype(np.float64) - ref[o])
            scale = float(np.abs(ref[o]).max()) or 1.0
            s = stats[o]
            s["max_abs"] = max(s["max_abs"], float(diff.max()))
            s["max_rel"] = max(s["max_rel"], float(diff.max()) / scale)
            s["sum_abs"] += float(diff.mean())
            s["count"] += 1
        head = out_names[0]
        a, b = (
            ours[head].reshape(-1, ours[head].shape[-1]),
            ref[head].reshape(-1, ref[head].shape[-1]),
        )
        top_ours, top_ref = a.argmax(-1), b.argmax(-1)
        agree += int((top_ours == top_ref).sum())
        total_positions += top_ref.size
        if vision:
            correct_mgc += int(top_ours[0] == label)
            correct_ort += int(top_ref[0] == label)
        if reference_graph is not None and i % interp_every == 0:
            got = run_graph(reference_graph, feeds)[head]
            interp["inputs"] += 1
            interp["max_abs_vs_ort"] = max(
                interp["max_abs_vs_ort"], float(np.abs(got - ref[head]).max())
            )
            t = got.reshape(-1, got.shape[-1]).argmax(-1)
            interp["top1_agree"] += int((t == top_ref).sum())
            interp["positions"] += t.size
        n += 1
        if n % 500 == 0:
            print(f"  {name}: {n} inputs", flush=True)
    result: dict[str, Any] = {
        "inputs": n,
        "input_kind": "imagenette2-320 val images" if vision else f"random token ids, seq {SEQ}",
        "compile_seconds": round(compile_seconds, 2),
        "compile_cached": compiled.report.cached if compiled.report else None,
        "outputs": {
            o: {
                "max_abs_err": s["max_abs"],
                "max_rel_err": s["max_rel"],
                "mean_abs_err": s["sum_abs"] / max(s["count"], 1),
            }
            for o, s in stats.items()
        },
        "top1_agreement": agree / max(total_positions, 1),
        "top1_positions": total_positions,
        "interpreter_O0": interp,
    }
    if vision:
        result["top1_accuracy_mgc"] = correct_mgc / n
        result["top1_accuracy_ort"] = correct_ort / n
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["resnet50", "mobilenetv2", "bert-base", "gpt2"])
    ap.add_argument(
        "--limit", type=int, default=None, help="images per vision model (all if unset)"
    )
    ap.add_argument("--text-inputs", type=int, default=50)
    ap.add_argument("--interp-every", type=int, default=100, help="0 disables the interpreter")
    ap.add_argument("--name", default="parity")
    args = ap.parse_args()
    results = {}
    for name in args.models:
        print(name, flush=True)
        start = time.perf_counter()
        results[name] = evaluate(name, args.limit, args.text_inputs, args.interp_every)
        results[name]["seconds"] = round(time.perf_counter() - start, 1)
        print({k: v for k, v in results[name].items() if k != "outputs"}, flush=True)
    path = write_result(
        args.name, {"models": results, "peak_rss_mb": round(peak_rss_mb(), 1), "args": vars(args)}
    )
    print("wrote", path)


if __name__ == "__main__":
    main()
    exit_now()
