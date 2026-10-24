#!/usr/bin/env bash
# Fresh NUMBA_CACHE_DIR: numba's cache=True keys on the caller's source file only,
# so a changed callee in another file can be masked by a stale cache entry.
# The path lives in a script, not an inline hook command, so a TMPDIR containing
# quotes, `$` or backticks is never re-parsed by a shell.
set -u

cache=$(mktemp -d) || exit 1
trap 'rm -rf -- "$cache"' EXIT

SPECTRA_JIT=1 NUMBA_CACHE_DIR="$cache" uv run --extra dev python -m pytest \
    tests/regression/ tests/unittest/ tests/numba/test_Feautrier_jit.py \
    -q -p no:cacheprovider "$@"
