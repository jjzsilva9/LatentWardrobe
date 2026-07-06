# Paint3D environment setup (WSL2)

Reproducible notes for recreating the Paint3D dev environment on this lab machine after a wipe. See [[project-lab-environment]] context (in Claude's memory) for why this exists — this machine has already lost setup state once. **It happened again 2026-07-03** (WSL distro + conda env + model caches all lost; host NVIDIA driver and WSL platform survived). The repo now lives on the U: network share (`\\tholospg.itserv.scss.tcd.ie\pgrad\silvajo`, i.e. `U:\My Documents\LatentWardrobe`), which survives wipes — recovery is: reinstall a distro, rerun the install script, re-download model weights on first run.

## Repo on U: — two extra gotchas (post-wipe 2026-07-03)

1. **WSL does not automount network drives.** Every WSL entry point must mount U: first:
   ```bash
   mkdir -p /mnt/u; mountpoint -q /mnt/u || mount -t drvfs U: /mnt/u
   ```
   The repo is then at `/mnt/u/My Documents/LatentWardrobe` (space in the path — always quote it in shell). The run scripts' launch one-liners include this mount guard.
2. **CRLF line endings.** Windows git here has `core.autocrlf=true`, so a fresh checkout gave every `.sh` CRLF endings — bash in WSL fails with `set: pipefail: invalid option name`. Fixed permanently via `.gitattributes` (`*.sh text eol=lf`); if a script ever fails like that again, `sed -i 's/\r$//' <file>`.

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

As of the 2026-07-03 rebuild: distro is **Ubuntu 26.04 LTS** installed via plain `wsl --install Ubuntu`, default user `root`; GPU is an NVIDIA RTX A4000 (host driver 581.15, CUDA 13.0 capable). The conda env only needs CUDA 11.3 at the toolkit level — the newer host driver is backward compatible. Note `nvidia-smi` inside WSL lives at `/usr/lib/wsl/lib/nvidia-smi` (not necessarily on PATH with `wsl -e`).

## Running the setup

From Windows PowerShell (mounts U: first, runs as root):
```powershell
wsl -u root bash -c 'mkdir -p /mnt/u; mountpoint -q /mnt/u || mount -t drvfs U: /mnt/u; cd "/mnt/u/My Documents/LatentWardrobe" && bash .claude/setup/install_paint3d_env.sh'
```

Takes a while — it's downloading ~2GB+ of PyTorch/CUDA packages plus kaolin/pytorch3d wheels.

## Gotchas hit during setup (all handled by the script, documented here for context)

1. **Conda ToS error**: newer conda versions refuse to solve against the default `pkgs/main`/`pkgs/r` channels until you explicitly accept their Terms of Service (`conda tos accept ...`). Silent failure otherwise with `CondaToSNonInteractiveError`.
2. **Missing `libgomp1`**: this WSL Ubuntu image is minimal and doesn't ship the OpenMP runtime torch's compiled libraries need. `import torch` fails with `OSError: libgomp.so.1: cannot open shared object file`.
3. **Executable-stack ELF error**: `import torch` failed with `cannot enable executable stack as shared object requires: Invalid argument`. This distro's glibc/kernel is new enough to enforce hardening that rejects old (2022-era) shared libraries requesting an executable stack — a harmless build artifact from how PyTorch 1.12.1 was originally compiled. Neither the conda-channel build nor the official pip wheel avoided this. Fix: `patchelf --clear-execstack` on torch's `.so` files. This is safe — it doesn't change what the library does, just removes a stale flag.
4. **kaolin wheel URL**: must match torch/CUDA version exactly via NVIDIA's own wheel index: `https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-1.12.1_cu113.html`.
5. **pytorch3d + `--no-index`**: the prebuilt-wheel install command from pytorch3d's docs uses `--no-index -f <fbaipublicfiles URL>`, which blocks pip from reaching PyPI at all — so it fails to resolve `iopath`/`fvcore` (pytorch3d's own deps), which aren't hosted on that link-only index. Fix: install `fvcore`/`iopath` normally from PyPI first, then install pytorch3d from the fbaipublicfiles wheel index without `--no-index`.
6. **`huggingface_hub`/`diffusers` version skew**: `environment.yaml` pins `diffusers==0.25.0` but not `huggingface_hub`, so pip installs whatever's newest. Newer `huggingface_hub` releases dropped `cached_download`, which diffusers 0.25.0 still imports at startup — `ImportError: cannot import name 'cached_download'`. Fix: pin `huggingface_hub==0.20.3` (from the same era as diffusers 0.25.0).
7. **`libcuda.so: cannot open shared object file` during actual pipeline runs** (not at `import torch` time — torch itself loads fine): cuDNN calls `dlopen("libcuda.so")` with no version suffix. WSL's `/usr/lib/wsl/lib` only has `libcuda.so.1` registered in the `ldconfig` cache — the unversioned `libcuda.so` symlink is treated as a dev-only file and `ldconfig` skips indexing it by design, even though the file itself is present in that directory. Fix: put `/usr/lib/wsl/lib` on `LD_LIBRARY_PATH` explicitly so it's found via directory search instead of the cache:
   ```bash
   export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
   ```
   Needed before running any `pipeline_paint3d_*.py` script, not just for importing torch.

## Verifying the environment

```bash
wsl -e bash -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate paint3d && python -c \"import torch, kaolin, pytorch3d; print(torch.__version__, torch.cuda.is_available(), kaolin.__version__, pytorch3d.__version__)\""
```

Expected (as of last verified run): `1.12.1+cu113 True 0.13.0 0.7.2`

## Running Paint3D itself

Verified end-to-end 2026-07-02 against the Suzanne monkey demo mesh (`external/paint3d/demo/objs/Suzanne_monkey/`) — both stage 1 (coarse, depth-conditioned) and stage 2 (UV inpaint + tile refinement) ran successfully and produced a fully-textured, seamless result (checked both the UV texture map and a rendered turntable frame). See `external/paint3d/README.md` for the pipeline commands; remember the `LD_LIBRARY_PATH` export above before running them.

Model checkpoints are pulled automatically from HuggingFace Hub on first run per the `sd_config` yaml files (`controlnet/config/*.yaml`) — no manual download needed. Note `runwayml/stable-diffusion-v1-5` (referenced by the configs) was taken down by Runway and now 307-redirects to `stable-diffusion-v1-5/stable-diffusion-v1-5`; `diffusers`/`huggingface_hub` follow this transparently, no config changes needed. First run downloads ~7-8GB of weights (SD1.5 + 2-3 ControlNets depending on stage); expect an occasional `ChunkedEncodingError`/`IncompleteRead` from a flaky connection mid-download — just rerun, already-downloaded files are cached.

Outputs land in `external/paint3d/outputs/` (gitignored-worthy — not committed). Stage 1's `res-0/albedo.png` will have magenta gaps where no camera view saw that UV region; that's expected and is what stage 2's UV inpainting fills in (final result: `outputs/stage2/tile_res_0/albedo.png`).

## Ollama (Method 5 — LLM emitter-mutation prompting)

`scripts/llm_prompt_mutation.py` calls a locally-served LLM via Ollama's HTTP API (`127.0.0.1:11434`), not a paid API — no key needed. Install with `.claude/setup/install_ollama.sh` (run as root — see that script's header for why, and for the zstd gotcha the stock installer hits on this minimal Ubuntu image). Pulls `llama3.1:8b` (~4.9GB, fits comfortably alongside Paint3D's SD1.5 stack on the RTX A4000's 16GB since the two never run concurrently — Method 5's LLM calls all happen before Paint3D's subprocesses launch per member). Runs as a systemd service (`ollama.service`), auto-starts on WSL boot; verify with `wsl -e bash -lc "systemctl status ollama"` or `curl 127.0.0.1:11434/api/tags`.
