# Overnight baseline run — launched 2026-07-02 evening

**POST-MORTEM (2026-07-03 morning): the 2026-07-02 launch produced nothing.** Two independent causes:
1. **Driver bug (the substantive one):** Paint3D stages run as subprocesses with `cwd=external/paint3d`, but the driver passed *relative* paths for configs/meshes/outdirs — every stage failed instantly with FileNotFoundError. Fixed 2026-07-03 (`args.outdir/mesh_dir` now resolved to absolute at startup).
2. **Launch method:** `wsl -e bash -c "nohup ... &"` does not survive — WSL2 tears down the VM shortly after the launching `wsl.exe` exits, killing the process (run.log was never even created). Detached runs must keep a `wsl.exe` client alive on the Windows side — use the Start-Process method below.

**What's running (relaunched 2026-07-03):** `scripts/generate_crowd_baseline.py`, 4 prompts x 25 members = 100 outfits, sequential. Estimated 8-13h total. Log: `run.log` (this folder).

Prompts: `superhero costume` / `aliens on the moon` / `medieval villagers` / `business casual office workers`

## Where to look in the morning

- Per-prompt manifests: `<prompt_slug>/manifest.json` — per-member status (`ok` / `FAILED rc=...` / `MERGE FAILED`), garments, seeds, timings.
- Final texture per member: `<prompt_slug>/member_XX/stage2/tile_res_0/albedo.png`
- Turntable render per member (easiest visual check): `<prompt_slug>/member_XX/stage2/tile_res_0/render_rgb.mp4`
- Stage-1-only (coarse) result if stage 2 failed: `member_XX/stage1/res-0/albedo.png`
- The merged input mesh: `member_XX/outfit.obj` (top in left half of UV atlas, bottom in right half)

Quick health check:

```bash
grep -c '"status": "ok"' */manifest.json          # completed stages
grep -l 'FAILED' */manifest.json                  # prompts with failures
tail -5 run.log                                   # is it still going / how far
```

## If it stopped partway

Relaunch via a detached `wsl.exe` (from PowerShell) — opens a VISIBLE console window streaming the full unfiltered log live (also tee'd to `run.log`); `--resume` skips everything already finished:

```powershell
Start-Process wsl.exe -ArgumentList '-u','root','-e','bash','/mnt/c/Users/silvajo/LatentWardrobe/scripts/run_overnight.sh'
```

Closing that console window kills the run (rerun the command to continue). Do NOT use `wsl -e bash -c "nohup ... &"` — see post-mortem above.
