# Model checkpoint (first-run)

Keno’s residual search needs a trained U-Net at:

```text
ml-engine/checkpoints/unet_denoiser.pt
```

Weights are **not** in git (~2 MB, gitignored). The paper freeze
`2026-07-paper-v1` pins this SHA256:

```text
55ce7637e14dd3558d4e9ede025a5e42e1ca25048a715bd87eb4f0fd028cd49a
```

(also in [`docs/freeze/current/checkpoint.sha256`](freeze/current/checkpoint.sha256))

Without this file the ML engine starts with **random weights** and demos are
meaningless. `/health` reports `checkpoint_loaded` and whether the file matches
the freeze hash.

## Recommended: download the freeze weights

Once the freeze `.pt` is published as a GitHub Release asset (or Zenodo file):

```bash
cd ml-engine
./scripts/fetch_checkpoint.sh
```

Or with an explicit URL:

```bash
CHECKPOINT_URL='https://github.com/PanosM397/Keno/releases/download/paper-v1/unet_denoiser.pt' \
  ./scripts/fetch_checkpoint.sh
```

Verify only:

```bash
./scripts/verify_checkpoint.sh
```

## Maintainer: publish the freeze `.pt` (one-time)

The freeze hash was recorded on the Windows machine that ran the paper freeze.
If `unet_denoiser.pt` with SHA `55ce7637…` still exists there (or in backup):

1. Copy it to `ml-engine/checkpoints/unet_denoiser.pt`
2. Confirm: `./scripts/verify_checkpoint.sh`
3. Create a GitHub Release tagged `paper-v1` and attach `unet_denoiser.pt`
4. Set the default URL in `ml-engine/scripts/fetch_checkpoint.sh` if needed
5. Optional: add the same file to a Zenodo **New version** of the preprint

Until that asset is online, `fetch_checkpoint.sh` will fail with a clear message.

## Alternative: train your own (not paper-reproducible)

```bash
cd ml-engine
source .venv/bin/activate
# see ml-engine/README.md — requires GWOSC downloads; can take hours
python -m app.training.train
```

Your local SHA will **not** match the freeze. Demos may still run, but they
will not match published GW150914 / freeze numbers. Prefer the freeze file for
portfolio and paper claims.

## Honest note on this workspace

As of the first-run docs update, the local `unet_denoiser.pt` on the Mac
checkout did **not** match `55ce7637…` (it had drifted after later retrain
attempts). Recover the freeze file before claiming live demos match
`2026-07-paper-v1`.
