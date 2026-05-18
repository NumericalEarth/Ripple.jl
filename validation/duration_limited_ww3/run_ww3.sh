#!/usr/bin/env bash
# Run the WW3 reference side of the Ripple duration-limited comparison.
#
# Prerequisites: WW3 binaries at $WW3_BIN (default /tmp/ww3-build/build/bin).
# Outputs land in output/ww3/ alongside the Ripple side.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WW3_BIN="${WW3_BIN:-/tmp/ww3-build/build/bin}"
OUT="${DIR}/output/ww3"
mkdir -p "$OUT"

# Stage input decks under fixed filenames WW3 expects.
cp "${DIR}/ww3_grid.inp" "$OUT/ww3_grid.inp"
cp "${DIR}/ww3_strt.inp" "$OUT/ww3_strt.inp"
cp "${DIR}/ww3_shel.inp" "$OUT/ww3_shel.inp"
cp "${DIR}/ww3_ounp.inp" "$OUT/ww3_ounp.inp"

cd "$OUT"

echo "── ww3_grid ────────────────────────────────────────────────────────"
"$WW3_BIN/ww3_grid"

echo "── ww3_strt ────────────────────────────────────────────────────────"
"$WW3_BIN/ww3_strt"

echo "── ww3_shel ────────────────────────────────────────────────────────"
"$WW3_BIN/ww3_shel"

echo "── ww3_ounp ────────────────────────────────────────────────────────"
"$WW3_BIN/ww3_ounp"

echo
echo "WW3 done. Output spectra in $OUT (ww3.point01.nc or similar)."
