#!/usr/bin/env bash
set -euo pipefail

RTKLIB_COMMIT="180043ee24b6d2b168f98b64be15f69d50046b1a"
DESTINATION="${1:-.tools/RTKLIB}"

if [[ -d "${DESTINATION}/.git" ]]; then
    printf '%s\n' "Reusing existing RTKLIB checkout at ${DESTINATION}"
else
    git clone https://github.com/tomojitakasu/RTKLIB.git "${DESTINATION}"
fi
git -C "${DESTINATION}" checkout --detach "${RTKLIB_COMMIT}"
make -C "${DESTINATION}/lib/iers/gcc"
make -C "${DESTINATION}/app/consapp/rnx2rtkp/gcc"

printf '%s\n' "Built ${DESTINATION}/app/consapp/rnx2rtkp/gcc/rnx2rtkp"
