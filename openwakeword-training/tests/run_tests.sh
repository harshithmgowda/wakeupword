#!/usr/bin/env bash
# Run the equivalence/correctness guards. See tests/README.md for what each
# one needs -- many require ./setup_env.sh to have been run, real recordings,
# or a trained model, and will fail loudly rather than silently skip.
#
#   ./tests/run_tests.sh            # everything
#   ./tests/run_tests.sh --light    # only tests with no heavy prerequisites
set -uo pipefail
cd "$(dirname "$0")/.."

# Tests that need nothing beyond numpy/scipy/soundfile.
LIGHT=(test_media_utils.py test_speech_fx.py test_make_tv_mix.py)

if [ -x .venv/bin/python ]; then
    PY=./.venv/bin/python
else
    PY=$(command -v python3)
    echo "note: .venv not found, falling back to $PY -- most tests will fail."
    echo "      run ./setup_env.sh first."
    echo
fi

if [ "${1:-}" = "--light" ]; then
    TESTS=("${LIGHT[@]}")
else
    TESTS=()
    for f in tests/test_*.py; do TESTS+=("$(basename "$f")"); done
fi

pass=0 fail=0 failed=()
for t in "${TESTS[@]}"; do
    printf '%-42s ' "$t"
    if out=$("$PY" "tests/$t" 2>&1); then
        echo "PASS"
        pass=$((pass + 1))
    else
        echo "FAIL"
        echo "$out" | tail -15 | sed 's/^/    /'
        fail=$((fail + 1))
        failed+=("$t")
    fi
done

echo
echo "$pass passed, $fail failed"
if [ "$fail" -gt 0 ]; then
    printf '  %s\n' "${failed[@]}"
    exit 1
fi
