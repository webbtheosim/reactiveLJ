# Custom HOOMD source

The original `matthewchertok/conservative-reactiveLJ` fork is private. This
directory provides the custom runtime source used by the public simulation
workflows without requiring access to that fork.

`reactivelj.bundle` contains two source-only release commits on top of public
[HOOMD-blue commit 8267227d29a570a27bf16d540fd193e5623dea97](https://github.com/glotzerlab/hoomd-blue/commit/8267227d29a570a27bf16d540fd193e5623dea97).
The runtime and build source matches the recorded study versions byte for
byte. Private commit history, local build caches, and account metadata are
excluded, so the public release has new commit IDs.

| Selection | Recorded study commit | Public source commit |
| --- | --- | --- |
| `revision` | `485771cd90b48b896f1fbb7cec35a8b14c9116ee` | `b4c0ad708bfc4ce1ca6aa0b4692b5f8f53b0c395` |
| `repository` | `90954a78fe2b999e8ba44b891e2f3e641d70d753` | `086d74827fa520b688e796456e0fc1184e7563dc` |

The revision launchers accept the original revision build or its specific
public source equivalent. Historical data retain their original provenance;
new simulations record the actual public build ID.

From the root of the ReactiveLJ repository, choose one source checkout:

```bash
# ReactiveAH comparison and smoothing sweep.
bash source_dependencies/hoomd/checkout.sh revision ../hoomd-revision-source

# Or the runtime source corresponding to the repository's HOOMD pin.
bash source_dependencies/hoomd/checkout.sh repository ../hoomd-public-source
```

The destination must be new or empty. The script downloads public upstream
HOOMD, verifies and imports the bundle, checks out the requested source,
and initializes HOOMD's public dependencies. It does not build or install
anything. Follow the [GPU build instructions](../../README.md#build-the-custom-hoomd-blue)
using the chosen source directory and a separate Python environment for
the revision campaigns.

The historical `.gitmodules` URL and original submodule pin are retained.
Use this setup script instead of recursively cloning the parent repository.
The public source is prepared outside the parent checkout because its new
commit ID differs from that historical pin. Within the prepared HOOMD
checkout, `git submodule update --init --recursive` downloads its public
nested dependencies normally.

Only `hoomd/md` source and build files change relative to upstream. HOOMD's
BSD 3-Clause [license](LICENSE) accompanies this source.
