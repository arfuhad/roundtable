"""Codebase scanning: turn an existing project into a compact text "digest".

The digest is the provider-agnostic input to the ``map`` command's analyst: a
pruned file tree plus the truncated contents of the files that explain a project
fastest (README, package manifests, entry points). It deliberately stays within a
byte budget so it fits in one LLM call, and is stdlib-only (no extra deps).

With ``provider: cli`` the analyst CLI agent additionally has live file access
(it runs with ``cwd`` set to the scanned project), so the digest is a reliable
floor, not a ceiling.
"""

from __future__ import annotations

from pathlib import Path

# Directories we never descend into: VCS, caches, deps, build output, IDE noise.
IGNORE_DIRS = frozenset({
    ".git", ".hg", ".svn", ".harness",
    "node_modules", ".venv", "venv", "env", "__pycache__",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", ".tox", ".cache",
    "dist", "build", "target", ".next", "out", "coverage", ".nyc_output",
    ".idea", ".vscode", "vendor", ".gradle", ".dart_tool",
})

# Files we list in the tree but never read (binary/asset/lock noise).
SKIP_READ_EXTS = frozenset({
    # images / media
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".ico", ".bmp",
    ".mp4", ".mov", ".webm", ".mp3", ".wav", ".ogg",
    # fonts
    ".ttf", ".otf", ".woff", ".woff2", ".eot",
    # archives / binaries
    ".zip", ".tar", ".gz", ".tgz", ".bz2", ".7z", ".rar",
    ".pdf", ".so", ".dylib", ".dll", ".o", ".a", ".class", ".pyc",
    ".bin", ".exe", ".wasm", ".db", ".sqlite",
})
SKIP_READ_NAMES = frozenset({
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock",
    "uv.lock", "Cargo.lock", "Gemfile.lock", "composer.lock", "go.sum",
})

# Manifest -> stack label, also the manifests we always inline.
_STACK_MANIFESTS: list[tuple[str, str]] = [
    ("pyproject.toml", "Python"),
    ("setup.py", "Python"),
    ("requirements.txt", "Python"),
    ("package.json", "Node"),
    ("Cargo.toml", "Rust"),
    ("go.mod", "Go"),
    ("pom.xml", "JVM"),
    ("build.gradle", "JVM"),
    ("build.gradle.kts", "JVM"),
    ("Gemfile", "Ruby"),
    ("composer.json", "PHP"),
    ("pubspec.yaml", "Dart"),
]

# Entry-point filenames worth inlining when present anywhere shallow.
_ENTRYPOINT_NAMES = frozenset({
    "main.py", "__main__.py", "app.py", "cli.py", "manage.py",
    "index.js", "index.ts", "main.js", "main.ts", "main.go", "main.rs",
})


def detect_stack(root: Path) -> list[str]:
    """Best-effort language/stack labels from manifest files at the root."""
    found: list[str] = []
    for name, label in _STACK_MANIFESTS:
        if (root / name).exists() and label not in found:
            found.append(label)
    return found


def _should_read(path: Path) -> bool:
    return path.suffix.lower() not in SKIP_READ_EXTS and path.name not in SKIP_READ_NAMES


def _iter_files(root: Path, *, max_files: int) -> list[Path]:
    """Deterministically walk ``root``, pruning ignored dirs; cap at ``max_files``."""
    out: list[Path] = []
    stack: list[Path] = [root]
    while stack:
        d = stack.pop()
        try:
            entries = sorted(d.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        except (PermissionError, OSError):
            continue
        dirs: list[Path] = []
        for e in entries:
            if e.is_symlink():
                continue
            if e.is_dir():
                if e.name not in IGNORE_DIRS:
                    dirs.append(e)
            elif e.is_file():
                out.append(e)
                if len(out) >= max_files:
                    return out
        # Push dirs reversed so the sorted order is preserved on pop (DFS).
        stack.extend(reversed(dirs))
    return out


def _key_files(root: Path, files: list[Path]) -> list[Path]:
    """Order the files most worth inlining first: README, manifests, entrypoints."""
    manifest_names = {name for name, _ in _STACK_MANIFESTS}
    readmes, manifests, entrypoints = [], [], []
    for f in files:
        if not _should_read(f):
            continue
        name = f.name
        if name.lower().startswith("readme"):
            readmes.append(f)
        elif name in manifest_names:
            manifests.append(f)
        elif name in _ENTRYPOINT_NAMES:
            entrypoints.append(f)
    # README first, then manifests, then a handful of entrypoints.
    return readmes + manifests + entrypoints


def _read_text(path: Path, limit: int) -> str:
    try:
        data = path.read_text(encoding="utf-8", errors="replace")
    except (OSError, ValueError):
        return ""
    if len(data) > limit:
        return data[:limit] + "\n...[truncated]..."
    return data


def build_digest(
    root: Path,
    *,
    max_files: int = 400,
    max_bytes_per_file: int = 8000,
    max_total_bytes: int = 60000,
) -> str:
    """Build a compact Markdown digest of ``root`` for the analyst.

    Sections: a stack line, a pruned file tree (capped at ``max_files``), and the
    inlined contents of key files (README/manifests/entrypoints), each truncated
    to ``max_bytes_per_file`` and the whole inline section bounded by
    ``max_total_bytes``.
    """
    root = Path(root).resolve()
    files = _iter_files(root, max_files=max_files)
    stack = detect_stack(root) or ["unknown"]

    lines: list[str] = [
        f"# Codebase digest: {root.name}",
        "",
        f"## Stack: {', '.join(stack)}",
        f"_{len(files)} files scanned (cap {max_files})._",
        "",
        "## File tree",
        "",
    ]
    for f in files:
        try:
            rel = f.relative_to(root).as_posix()
            size = f.stat().st_size
        except (OSError, ValueError):
            continue
        lines.append(f"- {rel} ({size}B)")
    if len(files) >= max_files:
        lines.append(f"- …(truncated at {max_files} files)")

    lines += ["", "## Key files", ""]
    budget = max_total_bytes
    inlined = 0
    for f in _key_files(root, files):
        if budget <= 0:
            lines.append("_…(content budget exhausted; remaining key files omitted)_")
            break
        text = _read_text(f, min(max_bytes_per_file, budget))
        if not text.strip():
            continue
        rel = f.relative_to(root).as_posix()
        lines += [f"### {rel}", "", "```", text, "```", ""]
        budget -= len(text)
        inlined += 1
    if inlined == 0:
        lines.append("_(no README, manifest, or entry point found to inline)_")

    return "\n".join(lines)
