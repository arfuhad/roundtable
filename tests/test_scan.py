"""Codebase digest: pruning, key-file inlining, stack detection, budgets."""

from harness.scan import build_digest, detect_stack


def _make_project(root):
    (root / "README.md").write_text("# Demo\n\nA sample project for scanning.\n")
    (root / "pyproject.toml").write_text("[project]\nname = \"demo\"\nversion = \"0.0.1\"\n")
    (root / "src" / "pkg").mkdir(parents=True)
    (root / "src" / "pkg" / "__init__.py").write_text("VALUE = 1\n")
    (root / "src" / "pkg" / "main.py").write_text("def main():\n    return VALUE\n")
    # noise that must be pruned / not read
    (root / "node_modules").mkdir()
    (root / "node_modules" / "junk.js").write_text("module.exports = 42;\n")
    (root / ".git").mkdir()
    (root / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    (root / "uv.lock").write_text("# lockfile\n" + "x = 1\n" * 100)


def test_detect_stack(tmp_path):
    _make_project(tmp_path)
    assert detect_stack(tmp_path) == ["Python"]


def test_digest_includes_key_files_and_prunes_noise(tmp_path):
    _make_project(tmp_path)
    digest = build_digest(tmp_path)

    # key files inlined
    assert "A sample project for scanning." in digest
    assert 'name = "demo"' in digest
    # tree lists source but excludes ignored dirs
    assert "src/pkg/main.py" in digest
    assert "node_modules" not in digest
    assert ".git" not in digest
    # lockfile may be listed in the tree but its contents are never inlined
    assert "# lockfile" not in digest
    assert "## Stack: Python" in digest


def test_digest_respects_file_cap(tmp_path):
    for i in range(20):
        (tmp_path / f"f{i:02d}.txt").write_text("x")
    digest = build_digest(tmp_path, max_files=5)
    assert "truncated at 5 files" in digest


def test_digest_respects_byte_budget(tmp_path):
    (tmp_path / "README.md").write_text("A" * 5000)
    (tmp_path / "pyproject.toml").write_text("B" * 5000)
    digest = build_digest(tmp_path, max_bytes_per_file=400, max_total_bytes=500)
    # per-file truncation kicks in
    assert "[truncated]" in digest
    # global budget keeps the whole digest small
    assert len(digest) < 2000
