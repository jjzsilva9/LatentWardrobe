"""Generate lightweight, GitHub-viewable previews of a generation batch.

Used by scripts/auto_push_results.ps1: while an overnight run is writing
outputs/<batch>/, this script distills each crowd into small JPEGs + a
per-crowd contact sheet + a README.md (GitHub renders README.md in directory
listings), written into the `results` branch worktree. Full outputs never go
to git (see .gitignore) — only these previews, manifests, metrics and a log
tail, so a whole overnight batch stays ~tens of MB.

Runs in the paint3d conda env (needs PIL + cv2). Idempotent and tolerant of a
batch that is actively being written: partial/unreadable files are skipped
and picked up on the next cycle.

    python scripts/make_previews.py \
        --batch "/mnt/u/My Documents/LatentWardrobe/outputs/shakedown2_20260703" \
        --dst   "/mnt/u/My Documents/LatentWardrobe-results/shakedown2_20260703"
"""

import argparse
import json
import math
import shutil
from pathlib import Path

import cv2
from PIL import Image, ImageDraw

THUMB = 512          # px, longest side of per-member previews
TILE = 256           # px, contact-sheet tile
LABEL_H = 32         # px, caption strip under each tile
LOG_TAIL_BYTES = 200_000

STAGE2_ALBEDO = "stage2/tile_res_0/albedo.png"
STAGE1_ALBEDO = "stage1/res-0/albedo.png"
TURNTABLE = "stage2/tile_res_0/render_rgb.mp4"


def newer(src: Path, dst: Path) -> bool:
    return not dst.exists() or src.stat().st_mtime > dst.stat().st_mtime


def save_thumb(src: Path, dst: Path, size=THUMB) -> bool:
    """PNG -> downscaled JPEG. Returns False on partial/unreadable source."""
    try:
        img = Image.open(src)
        img.load()
    except Exception:
        return False
    img = img.convert("RGB")
    img.thumbnail((size, size))
    dst.parent.mkdir(parents=True, exist_ok=True)
    img.save(dst, quality=85)
    return True


def grab_frames(mp4: Path, dst_dir: Path):
    """Extract front (frame 0) and back (middle frame) turntable views.
    A video still being written by ffmpeg has no moov atom yet -> cv2 can't
    open it -> returns [] and we retry next cycle."""
    outs = []
    try:
        cap = cv2.VideoCapture(str(mp4))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if n <= 0:
            cap.release()
            return []
        for label, idx in (("front", 0), ("back", n // 2)):
            dst = dst_dir / f"view_{label}.jpg"
            if not newer(mp4, dst):
                outs.append(dst)
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                continue
            h, w = frame.shape[:2]
            scale = THUMB / max(h, w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
            dst_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(dst), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            outs.append(dst)
        cap.release()
    except Exception:
        return outs
    return outs


def contact_sheet(tiles, dst: Path):
    """tiles: list of (image_path, caption). Grid JPEG with caption strips."""
    if not tiles:
        return
    cols = min(5, max(1, math.ceil(math.sqrt(len(tiles)))))
    rows = math.ceil(len(tiles) / cols)
    cell_h = TILE + LABEL_H
    sheet = Image.new("RGB", (cols * TILE, rows * cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (img_path, caption) in enumerate(tiles):
        r, c = divmod(i, cols)
        x, y = c * TILE, r * cell_h
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            continue
        img.thumbnail((TILE, TILE))
        sheet.paste(img, (x + (TILE - img.width) // 2, y + (TILE - img.height) // 2))
        draw.text((x + 4, y + TILE + 4), caption[:40], fill="black")
    dst.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(dst, quality=85)


def member_status(m: dict) -> str:
    for stage in ("stage2", "stage1", "outfit"):
        info = m.get(stage)
        if info:
            s = info.get("status", "?")
            return f"{stage}: {s}" if s != "ok" or stage != "stage2" else "ok"
    return "pending"


def process_crowd(crowd_dir: Path, dst: Path):
    manifest_path = crowd_dir / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text())
    except Exception:
        return None
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(manifest_path, dst / "manifest.json")
    if (crowd_dir / "metrics.json").exists():
        shutil.copy2(crowd_dir / "metrics.json", dst / "metrics.json")

    rows, sheet_tiles = [], []
    for m in manifest.get("members", []):
        mid = m["member"]
        mdir = crowd_dir / mid
        mdst = dst / mid
        albedo = next((mdir / p for p in (STAGE2_ALBEDO, STAGE1_ALBEDO)
                       if (mdir / p).exists()), None)
        thumb = None
        if albedo is not None:
            thumb = mdst / "albedo_thumb.jpg"
            if newer(albedo, thumb) and not save_thumb(albedo, thumb):
                thumb = None
        frames = grab_frames(mdir / TURNTABLE, mdst) if (mdir / TURNTABLE).exists() else []

        garments = " + ".join(Path(g).stem for g in m.get("garments", []))
        rows.append((mid, garments, m.get("sd_seed", "?"), member_status(m),
                     thumb, frames))
        face = frames[0] if frames else thumb
        if face is not None:
            sheet_tiles.append((face, f"{mid}  {garments}"))

    contact_sheet(sheet_tiles, dst / "contact_sheet.jpg")
    return manifest, rows


def write_readme(dst_root: Path, batch_name: str, crowds):
    lines = [f"# {batch_name} — live previews", "",
             "Auto-published while the batch runs; refresh for updates. "
             "Full outputs stay on the lab machine (regenerable from manifests).", ""]
    for slug, manifest, rows in crowds:
        done = sum(1 for r in rows if r[3] == "ok")
        lines += [f"## {manifest['crowd_prompt']}  ({done}/{len(rows)} members done)", ""]
        metrics_path = dst_root / slug / "metrics.json"
        if metrics_path.exists():
            try:
                d = json.loads(metrics_path.read_text())["diversity"]
                c = json.loads(metrics_path.read_text())["coherence"]
                lines += ["| metric | value |", "|---|---|",
                          f"| LPIPS pairwise mean / min-NN | {d['lpips_pairwise']['mean']:.4f} / {d['lpips_pairwise']['min']:.4f} |",
                          f"| CLIP distance mean / min-NN | {d['clip_distance_pairwise']['mean']:.4f} / {d['clip_distance_pairwise']['min']:.4f} |",
                          f"| Vendi score | {d['vendi_score_clip']:.2f} |",
                          f"| Theme coherence (crowd mean) | {c['clipscore_crowd_mean']:.4f} |", ""]
            except Exception:
                pass
        if (dst_root / slug / "contact_sheet.jpg").exists():
            lines += [f"![contact sheet]({slug}/contact_sheet.jpg)", ""]
        lines += ["| member | garments | seed | status | previews |", "|---|---|---|---|---|"]
        for mid, garments, seed, status, thumb, frames in rows:
            links = " ".join(f"[{p.stem}]({slug}/{mid}/{p.name})" for p in
                             ([thumb] if thumb else []) + list(frames))
            lines.append(f"| {mid} | {garments} | {seed} | {status} | {links} |")
        lines.append("")
    (dst_root / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch", type=Path, required=True, help="outputs/<batch> dir being generated")
    ap.add_argument("--dst", type=Path, required=True, help="destination dir inside the results worktree")
    args = ap.parse_args()

    crowds = []
    for crowd_dir in sorted(p for p in args.batch.iterdir() if p.is_dir()):
        if not (crowd_dir / "manifest.json").exists():
            continue
        res = process_crowd(crowd_dir, args.dst / crowd_dir.name)
        if res:
            crowds.append((crowd_dir.name, res[0], res[1]))

    log = args.batch / "run.log"
    if log.exists():
        data = log.read_bytes()[-LOG_TAIL_BYTES:]
        (args.dst / "run_tail.log").write_bytes(data)

    args.dst.mkdir(parents=True, exist_ok=True)
    write_readme(args.dst, args.batch.name, crowds)
    print(f"previews updated: {args.dst} ({len(crowds)} crowd(s))")


if __name__ == "__main__":
    main()
