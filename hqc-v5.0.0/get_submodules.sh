#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)
SUBMODULE_PATH="hqc-v5.0.0/next-release"

cd "${REPO_ROOT}"

git submodule update --init --recursive -- "${SUBMODULE_PATH}"
SUBMODULE_COMMIT=$(git -C "${SUBMODULE_PATH}" rev-parse HEAD)

echo "Submodule path: ${SUBMODULE_PATH}"
echo "Submodule commit: ${SUBMODULE_COMMIT}"
