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

## Maintainer: recover from Windows (one-time)

The freeze was recorded on **Windows-10** (`docs/freeze/current/MANIFEST.json`,
2026-07-17 UTC). The checkpoint was **not** copied into git or Zenodo — only
its SHA256 was pinned. Expected file size: **1 994 074 bytes**.

On the Windows machine where you ran `python -m app.evaluation.freeze_bundle`,
look for:

```text
<Keno>\ml-engine\checkpoints\unet_denoiser.pt
```

Verify before copying (PowerShell):

```powershell
Get-FileHash .\ml-engine\checkpoints\unet_denoiser.pt -Algorithm SHA256
# Hash must be: 55ce7637e14dd3558d4e9ede025a5e42e1ca25048a715bd87eb4f0fd028cd49a
```

Also check: File History, OneDrive, USB backups, old zip exports, or any clone
of Keno from **before** Jul 17 retrain on the Mac (Mac copies here are **wrong**
hash — `962ffc64…`, `be953501…`, `4870a0d6…`).

Copy the verified file to this Mac:

```bash
# after scp / AirDrop / cloud sync
cp /path/from/windows/unet_denoiser.pt ml-engine/checkpoints/
cd ml-engine && ./scripts/verify_checkpoint.sh
```

## Maintainer: publish GitHub Release `paper-v1`

Once the verified file is at `ml-engine/checkpoints/unet_denoiser.pt`:

```bash
gh auth login   # once
cd ml-engine
./scripts/publish_checkpoint_release.sh
```

That script verifies SHA256, tags `paper-v1` (if missing), pushes the tag, and
uploads `unet_denoiser.pt` so `./scripts/fetch_checkpoint.sh` works for others.

Manual alternative:

1. `./scripts/verify_checkpoint.sh`
2. `git tag -a paper-v1 -m "Keno scientific paper freeze 2026-07-paper-v1"`
3. `git push origin paper-v1`
4. GitHub → Releases → **paper-v1** → attach `unet_denoiser.pt`

Optional: add the same file to a Zenodo **New version** of the preprint.

Until that asset is online, `fetch_checkpoint.sh` returns **404** with a hint.

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
