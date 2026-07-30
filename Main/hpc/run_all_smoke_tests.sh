#!/usr/bin/env bash
#
# run_all_smoke_tests.sh
# =======================
# Correct invocation of Smoke_Tests/run_all_smoke_tests.py.
#
# WHY THIS WRAPPER EXISTS
# ------------------------
# The documented command
#
#     cd Main && PYTHONPATH=. python3 Smoke_Tests/run_all_smoke_tests.py
#
# is WRONG for this specific runner, and it fails silently-ish (as
# ModuleNotFoundError inside individual suites, not as an obvious top-level
# error). run_all_smoke_tests.py spawns each suite as a SEPARATE subprocess
# with its own cwd set to Smoke_Tests/ (see subprocess.run(cmd, cwd=str(here))
# in the runner). A RELATIVE PYTHONPATH="." is resolved by each subprocess
# against ITS OWN cwd -- which is Smoke_Tests/, not Main/ -- so any suite that
# does not independently self-locate Main/ via __file__ raises
# "ModuleNotFoundError: No module named 'config'" (or 'backbone', etc.), while
# suites that do self-locate happen to pass regardless. This produces a
# confusing PARTIAL pass/fail pattern that looks like a real code problem but
# is purely an invocation issue.
#
# The fix is a one-word change: PYTHONPATH must be an ABSOLUTE path to Main/,
# which resolves identically no matter what cwd a spawned subprocess ends up
# in. That is all this script does.
#
# Usage (from anywhere):
#     bash run_all_smoke_tests.sh                 # run all 25 suites
#     bash run_all_smoke_tests.sh --quick          # pass --quick where accepted
#     bash run_all_smoke_tests.sh --only train      # substring-match suites
#
# Requires: python3 with torch, pytorch_metric_learning, scikit-optimize,
# scikit-learn, numpy, scipy installed (the cluster environment; this is NOT
# the pure-shell landing script from earlier -- Python is required here
# because the suites themselves are Python).

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# If this script sits at the repo root next to Main/, MAIN is Main/. If it
# sits inside Main/ itself, MAIN is HERE. Handle both without forcing the
# caller to remember which.
if [ -d "$HERE/Main" ]; then
  MAIN="$HERE/Main"
elif [ -f "$HERE/config.py" ] && [ -d "$HERE/Smoke_Tests" ]; then
  MAIN="$HERE"
else
  echo "ABORT: could not locate Main/ from $HERE. Run this from the repo root" >&2
  echo "       or from inside Main/ itself." >&2
  exit 1
fi

echo "Main/ resolved to: $MAIN"
echo "PYTHONPATH set to : $MAIN  (absolute -- required, see header comment)"
echo ""

cd "$MAIN"
PYTHONPATH="$MAIN" exec python3 -W ignore::RuntimeWarning \
  Smoke_Tests/run_all_smoke_tests.py "$@"
