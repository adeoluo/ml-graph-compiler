#!/usr/bin/env bash
# Cut a release from Conventional Commits, run locally.
#
#   scripts/release.sh             next version inferred from commits since the last tag
#   scripts/release.sh v1.2.3      explicit version
#   scripts/release.sh --dry-run   print the version and notes, change nothing
#
# Steps: bump the manifest version, regenerate CHANGELOG.md, commit, tag, push,
# build release assets when scripts/build-assets.sh exists, publish the GitHub release.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

dry_run=false
version=""
for arg in "$@"; do
  case "$arg" in
    --dry-run) dry_run=true ;;
    v[0-9]*) version="$arg" ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

for tool in git-cliff gh; do
  command -v "$tool" >/dev/null || { echo "missing required tool: $tool" >&2; exit 1; }
done

git fetch --quiet --tags origin
current="$(git describe --tags --abbrev=0 --match 'v[0-9]*.[0-9]*.[0-9]*' 2>/dev/null || echo none)"
next="${version:-$(git cliff --bumped-version 2>/dev/null)}"
[[ "$next" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]] || { echo "not a semantic version: $next" >&2; exit 1; }
if [ "$next" = "$current" ]; then
  echo "no releasable commits since $current"
  exit 0
fi

notes="$(mktemp)"
trap 'rm -f "$notes"' EXIT
git cliff --unreleased --tag "$next" --strip all > "$notes"

if $dry_run; then
  echo "current: $current"
  echo "next:    $next"
  echo
  cat "$notes"
  exit 0
fi

[ "$(git rev-parse --abbrev-ref HEAD)" = main ] || { echo "release from main" >&2; exit 1; }
[ -z "$(git status --porcelain)" ] || { echo "working tree is not clean" >&2; exit 1; }
[ "$(git rev-parse HEAD)" = "$(git rev-parse origin/main)" ] || { echo "main differs from origin/main" >&2; exit 1; }

plain="${next#v}"
# Only the first top-level version key is the package version in these manifests.
for manifest in pyproject.toml Cargo.toml; do
  if [ -f "$manifest" ] && grep -q '^version = "' "$manifest"; then
    perl -0pi -e "s/^version = \"[^\"]*\"/version = \"$plain\"/m" "$manifest"
  fi
done
[ -f Cargo.lock ] && cargo metadata --format-version 1 --offline >/dev/null 2>&1 || true
[ -f uv.lock ] && uv lock --quiet 2>/dev/null || true
[ -f package.json ] && npm version "$plain" --no-git-tag-version --allow-same-version >/dev/null

git cliff --tag "$next" --output CHANGELOG.md
perl -0pi -e 's/\n+\z/\n/' CHANGELOG.md
git add CHANGELOG.md
for f in pyproject.toml Cargo.toml Cargo.lock uv.lock package.json package-lock.json; do
  [ -f "$f" ] && git add "$f"
done
git commit --quiet -m "chore(release): $next"
git tag -a "$next" -m "Release $next"
git push --quiet origin main "$next"

# GitHub Actions resolve uses: owner/repo@v1 through a moving major tag.
if [ -f action.yml ]; then
  major="${next%%.*}"
  git tag -fa "$major" -m "Track $next" "$next"
  git push --quiet --force origin "$major"
fi

assets=()
if [ -x scripts/build-assets.sh ]; then
  scripts/build-assets.sh "$next"
  for a in dist/release/*; do [ -f "$a" ] && assets+=("$a"); done
fi
gh release create "$next" --title "$next" --notes-file "$notes" --verify-tag "${assets[@]+"${assets[@]}"}"
echo "released $next"
