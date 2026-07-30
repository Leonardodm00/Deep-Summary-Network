"""
smoke_test_removed_modules.py

Deletion guard for Change 2 of the v3 handoff: the latent-factor RETENTION
metric (module C5) and its smoke test were deleted, and this suite asserts that
they stay deleted.

Why a test for an absence
-------------------------
A deletion leaves no code to exercise, so nothing else in the suite would notice
it being undone. The failure mode is undramatic -- someone restores the module
from an archived zip, a merge from an older branch brings it back, or a
docstring goes on telling a reader to run a file that no longer exists -- but
the end state is the one Change 2 removed: a repository that documents and
half-carries an analysis with no production consumer. This suite is the only
thing standing between that state and a green test run.

What it checks is deliberately slightly MORE than the handoff asks. Handoff
section 8.1 requires [A] the runner no longer names the deleted suite and still
runs, and [B] a reference scan over Main/. The file-level checks in group A are
what both of those presuppose, and the zip-membership scan in group C covers the
one place a plain text grep cannot look: the deployment archives that are copied
to the cluster.

Self-reference, deliberate
--------------------------
This file scans the tree for two forbidden tokens, so it must not CONTAIN them
or it would fail against itself. They are assembled from fragments at run time
(see BANNED below). That is not obfuscation and it is not to be "tidied" into
literals; it is what lets the assertion be stated without an exemption for the
file that states it.

Run:
    cd Main && PYTHONPATH=. python3 Smoke_Tests/smoke_test_removed_modules.py

Needs no torch, no data and no import from the package, which is why the runner
places it first.

Checks:
  A. The deleted module and the deleted suite are absent from disk, no compiled
     .pyc of the module survives in __pycache__, and the module is not
     importable from the package directory (an installed copy or a stale
     bytecode file would make the deletion cosmetic).
  B. run_all_smoke_tests.py --list exits 0 and names neither the deleted suite
     nor either forbidden token.
  C. Neither forbidden token occurs in any text file under Main/, nor in the
     member list of any .zip under Main/. The changelog that records the removal
     lives OUTSIDE Main/, which is the carve-out the handoff's assertion [B]
     allows.

HPC note (hpc-python-compat): pure ASCII.
"""

import importlib.util
import subprocess
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent          # Main/Smoke_Tests
MAIN = HERE.parent                              # Main

# Assembled at run time -- see "Self-reference, deliberate" above.
_TOK = "factor" + "_retention"
BANNED = (_TOK, "[C" + "5]")

DELETED_MODULE_STEM = _TOK                      # factor-retention module stem
DELETED_MODULE = MAIN / (DELETED_MODULE_STEM + ".py")
DELETED_SUITE = HERE / ("smoke_test_" + DELETED_MODULE_STEM + ".py")

# Directories never scanned, and suffixes that are not text.
SKIP_DIRS = {".git", "__pycache__", ".ipynb_checkpoints", ".mypy_cache",
             ".pytest_cache"}
BINARY_SUFFIXES = {".zip", ".png", ".jpg", ".jpeg", ".pdf", ".pt", ".pth",
                   ".npz", ".npy", ".pkl", ".pyc", ".so", ".ico"}


# --------------------------------------------------------------------------- #
# [A] the files are actually gone
# --------------------------------------------------------------------------- #
def check_files_absent():
    assert not DELETED_MODULE.exists(), (
        "%s is back on disk. It was deleted in Change 2 because it had no "
        "production consumer; if it is wanted again, that is a decision to "
        "make explicitly, not by restoring a file." % DELETED_MODULE)

    assert not DELETED_SUITE.exists(), (
        "%s is back on disk; its module is gone, so it cannot pass."
        % DELETED_SUITE)

    stale = sorted((MAIN / "__pycache__").glob(DELETED_MODULE_STEM + ".*.pyc")) \
        if (MAIN / "__pycache__").is_dir() else []
    assert not stale, (
        "the source is gone but compiled bytecode survives: %s. Python will "
        "import that happily and the deletion is cosmetic." % [str(p) for p in stale])

    # Not importable from the package directory either. sys.path is restored
    # so this check cannot perturb the ones that follow.
    saved = list(sys.path)
    try:
        sys.path.insert(0, str(MAIN))
        importlib.invalidate_caches()
        try:
            spec = importlib.util.find_spec(DELETED_MODULE_STEM)
        except (ImportError, ValueError):
            spec = None
        assert spec is None, (
            "the module is importable from %s (origin %r) even though the file "
            "was deleted -- an installed or vendored copy is shadowing the "
            "deletion." % (MAIN, getattr(spec, "origin", None)))
    finally:
        sys.path[:] = saved

    return ("module, suite and bytecode absent; not importable from %s/"
            % MAIN.name)


# --------------------------------------------------------------------------- #
# [B] the runner (handoff 8.1 assertion [A])
# --------------------------------------------------------------------------- #
def check_runner_clean():
    runner = HERE / "run_all_smoke_tests.py"
    assert runner.exists(), "run_all_smoke_tests.py is missing from %s" % HERE

    proc = subprocess.run([sys.executable, str(runner), "--list"],
                          cwd=str(HERE), capture_output=True, text=True)
    assert proc.returncode == 0, (
        "run_all_smoke_tests.py --list exited %d:\n%s\n%s"
        % (proc.returncode, proc.stdout[-800:], proc.stderr[-800:]))

    out = proc.stdout
    for token in BANNED:
        assert token not in out, (
            "run_all_smoke_tests.py --list still names %r; remove the entry "
            "from ORDER and from the description map." % token)
    assert DELETED_SUITE.name not in out, (
        "run_all_smoke_tests.py --list still names %s" % DELETED_SUITE.name)

    # The listing must not be empty, or the two assertions above would pass
    # vacuously on a broken runner.
    n_listed = sum(1 for line in out.splitlines()
                   if line.strip().startswith("smoke_test_"))
    assert n_listed >= 5, (
        "run_all_smoke_tests.py --list named only %d suite(s); the checks above "
        "would pass vacuously. Output was:\n%s" % (n_listed, out))

    return ("--list exits 0 and names %d suite(s), none of them the deleted one"
            % n_listed)


# --------------------------------------------------------------------------- #
# [C] the reference scan (handoff 8.1 assertion [B])
# --------------------------------------------------------------------------- #
def _text_files_under(root):
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.suffix.lower() in BINARY_SUFFIXES:
            continue
        yield path


def check_no_references():
    hits = []
    n_files = 0
    for path in _text_files_under(MAIN):
        n_files += 1
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            for token in BANNED:
                if token in line:
                    hits.append((path.relative_to(MAIN), lineno, token))

    assert not hits, (
        "%d dangling reference(s) to the deleted metric under Main/:\n%s\n"
        "The changelog outside Main/ is the only place these tokens belong."
        % (len(hits), "\n".join("  %s:%d  %s" % h for h in hits[:20])))

    # Plain text is not the only carrier: the deployment archives under Main/
    # are copied to the cluster and unpacked there.
    n_zips = 0
    zip_hits = []
    for path in sorted(MAIN.rglob("*.zip")):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        n_zips += 1
        try:
            with zipfile.ZipFile(path) as zf:
                names = zf.namelist()
        except (OSError, zipfile.BadZipFile):
            continue
        for name in names:
            for token in BANNED:
                if token in name:
                    zip_hits.append((path.relative_to(MAIN), name))

    assert not zip_hits, (
        "the deleted module is still shipped inside %d archive(s): %s. "
        "Rebuild them; a stale zip redeploys the file to the cluster."
        % (len(zip_hits), zip_hits[:10]))

    return ("%d text file(s) and %d archive(s) scanned, no reference found"
            % (n_files, n_zips))


def main():
    groups = [
        ("A", "deleted files absent and unimportable", check_files_absent),
        ("B", "run_all_smoke_tests.py --list is clean", check_runner_clean),
        ("C", "no reference under Main/, text or archived", check_no_references),
    ]
    print("smoke_test_removed_modules.py  [Change 2 deletion guard]")
    failures = []
    for letter, title, fn in groups:
        try:
            detail = fn()
        except Exception as ex:                    # noqa: BLE001
            failures.append((letter, title, ex))
            print("  [%s] %-44s FAIL" % (letter, title))
            print("      %s: %s" % (type(ex).__name__, ex))
        else:
            print("  [%s] %-44s PASS" % (letter, title))
            if detail:
                print("      %s" % detail)
    if failures:
        print("FAILED: %d of %d assertion group(s): %s"
              % (len(failures), len(groups),
                 ", ".join(f[0] for f in failures)))
        return 1
    print("ALL REMOVED-MODULE CHECKS PASSED (%d groups)" % len(groups))
    return 0


if __name__ == "__main__":
    sys.exit(main())
