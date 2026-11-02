#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 2 ]]; then
    echo "Usage: bash checkout.sh {revision|repository} NEW_SOURCE_DIRECTORY" >&2
    exit 2
fi

case "$1" in
    revision) source_commit=b4c0ad708bfc4ce1ca6aa0b4692b5f8f53b0c395 ;;
    repository) source_commit=086d74827fa520b688e796456e0fc1184e7563dc ;;
    *) echo "Choose revision or repository." >&2; exit 2 ;;
esac

bundle_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
destination=$2

# git clone accepts a new or empty directory and refuses to overwrite a checkout.
git clone --filter=blob:none --no-checkout \
    https://github.com/glotzerlab/hoomd-blue.git "$destination"
git -C "$destination" bundle verify "$bundle_dir/reactivelj.bundle"
git -C "$destination" fetch "$bundle_dir/reactivelj.bundle" \
    refs/heads/public-source:refs/remotes/reactivelj/public-source
git -C "$destination" checkout --detach "$source_commit"
git -C "$destination" submodule update --init --recursive

echo "Prepared HOOMD source at $(git -C "$destination" rev-parse HEAD)"
