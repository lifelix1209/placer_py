#!/usr/bin/env bash
# Regenerate the golden vectors from the C++ decision layer.
#
# Run this whenever the C++ decision layer changes ON PURPOSE. A diff in
# tests/oracle/cpp_reference.json is the signal that the migration contract
# moved, and it should never move by accident -- review the diff before
# committing it.
#
# This repository is standalone, so it has to be TOLD where the C++ lives:
#
#     PLACER_SRC=/path/to/PLACER tools/regenerate_oracle.sh
#
# and it defaults to a sibling checkout (../PLACER) because that is the layout
# the two repositories are developed in. Pointing it at the wrong tree would
# silently regenerate the contract from a different codebase, so the path is
# checked for `include/pipeline.h` before anything is compiled.
#
# Only the pure-logic sources are needed, which is why this does not require
# CMake, abPOA, or a built pipeline. It does need the htslib HEADERS, because
# include/pipeline.h pulls in include/bam_io.h. Headers only; nothing is linked
# against htslib.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/.." && pwd)"
SRC="${PLACER_SRC:-$(cd "${ROOT}/.." && pwd)/PLACER}"
OUT="${ROOT}/tests/oracle/cpp_reference.json"

if [[ ! -f "${SRC}/include/pipeline.h" ]]; then
  echo "error: no C++ checkout at ${SRC}" >&2
  echo "       set PLACER_SRC to the PLACER repository root" >&2
  exit 1
fi

HTS_INC="${HTSLIB_INCLUDE_DIR:-}"
EXTRA_INC=()
if [[ -n "${HTS_INC}" ]]; then
  EXTRA_INC+=("-I${HTS_INC}")
fi

echo "c++ sources: ${SRC}"
echo "output:      ${OUT}"

TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT

g++ -std=c++17 -O1 \
  "-I${SRC}/include" \
  "${EXTRA_INC[@]+"${EXTRA_INC[@]}"}" \
  "${HERE}/dump_oracle.cpp" \
  "${SRC}/src/pipeline/mechanistic_evidence.cpp" \
  "${SRC}/src/pipeline/decision_policy.cpp" \
  "${SRC}/src/pipeline/event_explanation.cpp" \
  "${SRC}/src/component/te_sequence_explainer.cpp" \
  -o "${TMP}/dump_oracle"

"${TMP}/dump_oracle" > "${TMP}/out.json"
python3 -c "import json,sys; json.load(open(sys.argv[1]))" "${TMP}/out.json"
mv "${TMP}/out.json" "${OUT}"

echo
echo "regenerated. Review the diff:"
echo "    git diff -- tests/oracle/cpp_reference.json"
echo
echo "If it is non-empty, the contract moved. Decide whether that was intended"
echo "before committing, and re-run the Python suite either way."
