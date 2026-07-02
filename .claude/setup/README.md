# Paint3D environment setup (WSL2)

Reproducible notes for recreating the Paint3D dev environment on this lab machine after a wipe. See [[project-lab-environment]] context (in Claude's memory) for why this exists — this machine has already lost setup state once.

## Why WSL2

Paint3D's own stack (`external/paint3d/environment.yaml`) is old and Linux-oriented: Python 3.8.5, CUDA 11.3, PyTorch 1.12.1, plus compiled-extension packages (`kaolin`, `pytorch3d`) that are painful to build on native Windows but have Linux wheels readily available. WSL2 isolates that old CUDA toolkit from the host NVIDIA driver, so it doesn't conflict with anything else on the machine.

## One-time host prerequisites (outside this script's scope)

- WSL2 installed with an Ubuntu distro (`wsl -l -v` should show it).
- NVIDIA driver on the Windows host with WSL CUDA support (check `nvidia-smi` on the host).
- Verify GPU passthrough into WSL **before** running the install script:
  ```
  wsl -e nvidia-smi
  ```
  Should print the GPU. If this fails, the issue is host driver / WSL kernel, not this repo's setup — fix that first.

As of the last full setup (2026-07-02), this machine's WSL Ubuntu distro's default user is `root`, and the GPU is an NVIDIA RTX A4000 (host driver 581.15, CUDA 13.0 capable). The conda env only needs CUDA 11.3 at the toolkit level — the newer host driver is backward compatible.

## Running the setup

From inside WSL:
```bash
cd /mnt/c/Users/silvajo/LatentWardrobe   # or wherever the repo lives
bash .claude/setup/install_paint3d_env.sh
```

Or from Windows PowerShell:
```powershell
wsl -e bash -c "cd /mnt/c/Users/silvajo/LatentWardrobe && bash .claude/setup/install_paint3d_env.sh"
```

Takes a while — it's downloading ~2GB+ of PyTorch/CUDA packages plus kaolin/pytorch3d wheels.

## Gotchas hit during setup (all handled by the script, documented here for context)

1. **Conda ToS error**: newer conda versions refuse to solve against the default `pkgs/main`/`pkgs/r` channels until you explicitly accept their Terms of Service (`conda tos accept ...`). Silent failure otherwise with `CondaToSNonInteractiveError`.
2. **Missing `libgomp1`**: this WSL Ubuntu image is minimal and doesn't ship the OpenMP runtime torch's compiled libraries need. `import torch` fails with `OSError: libgomp.so.1: cannot open shared object file`.
3. **Executable-stack ELF error**: `import torch` failed with `cannot enable executable stack as shared object requires: Invalid argument`. This distro's glibc/kernel is new enough to enforce hardening that rejects old (2022-era) shared libraries requesting an executable stack — a harmless build artifact from how PyTorch 1.12.1 was originally compiled. Neither the conda-channel build nor the official pip wheel avoided this. Fix: `patchelf --clear-execstack` on torch's `.so` files. This is safe — it doesn't change what the library does, just removes a stale flag.
4. **kaolin wheel URL**: must match torch/CUDA version exactly via NVIDIA's own wheel index: `https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-1.12.1_cu113.html`.
5. **pytorch3d + `--no-index`**: the prebuilt-wheel install command from pytorch3d's docs uses `--no-index -f <fbaipublicfiles URL>`, which blocks pip from reaching PyPI at all — so it fails to resolve `iopath`/`fvcore` (pytorch3d's own deps), which aren't hosted on that link-only index. Fix: install `fvcore`/`iopath` normally from PyPI first, then install pytorch3d from the fbaipublicfiles wheel index without `--no-index`.

## Verifying the environment

```bash
wsl -e bash -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate paint3d && python -c \"import torch, kaolin, pytorch3d; print(torch.__version__, torch.cuda.is_available(), kaolin.__version__, pytorch3d.__version__)\""
```

Expected (as of last verified run): `1.12.1+cu113 True 0.13.0 0.7.2`

## Running Paint3D itself

Not yet exercised end-to-end in this repo — next step is running `pipeline_paint3d_stage1.py` against a demo mesh to confirm the full pipeline (SD + ControlNet weights, etc.) works, not just the Python import stack. See `external/paint3d/README.md` for the pipeline commands.
