from pathlib import Path
from PIL import Image, ImageSequence
import re

# Input/output folders
DIR = Path("build/dce_7_mdreg/Bordeaux/Patients/_GIFS")
OUT = Path("build/dce_7_mdreg/Bordeaux/Patients/moco_gifs")
OUT.mkdir(exist_ok=True, parents=True)

# Match baseline 1 or 5 and moving/coreg
iter_re = re.compile(r"MDREG_(\d+)_(\d+)_baseline_(1|5)_(coreg|moving)\.gif$")

# Organize files: subject → baseline → type
files = {}
for f in DIR.glob("*.gif"):
    m = iter_re.search(f.name)
    if not m:
        continue
    study, subject, baseline, gtype = m.groups()
    key = f"{study}_{subject}"
    files.setdefault(key, {})[baseline, gtype] = f

for key, baselines in files.items():
    # Required GIFs: baseline 1 & 5, coreg & moving
    required = [("1", "moving"), ("1", "coreg"), ("5", "moving"), ("5", "coreg")]
    if any(r not in baselines for r in required):
        print(f"Skipping {key}, missing one of the required GIFs")
        continue

    # Open GIFs
    imgs = {r: Image.open(baselines[r]) for r in required}

    # Determine widths/heights
    col1_width = max(imgs["1","moving"].width, imgs["5","moving"].width)
    col2_width = max(imgs["1","coreg"].width, imgs["5","coreg"].width)
    row1_height = max(imgs["1","moving"].height, imgs["1","coreg"].height)
    row2_height = max(imgs["5","moving"].height, imgs["5","coreg"].height)

    # Number of frames: minimum across all GIFs
    n_frames = min(getattr(imgs[r], "n_frames", 1) for r in required)

    frames = []

    def resize_to(img, w, h):
        return img.resize((w, h))  # No antialiasing

    for i in range(n_frames):
        for r in required:
            imgs[r].seek(i)

        # Create new frame (2x1 grid)
        new_frame = Image.new("RGBA", (col1_width + col2_width, row1_height + row2_height))

        # Paste top row (baseline 1)
        new_frame.paste(resize_to(imgs["1","moving"].convert("RGBA"), col1_width, row1_height), (0,0))
        new_frame.paste(resize_to(imgs["1","coreg"].convert("RGBA"), col2_width, row1_height), (col1_width,0))

        # Paste bottom row (baseline 5)
        new_frame.paste(resize_to(imgs["5","moving"].convert("RGBA"), col1_width, row2_height), (0,row1_height))
        new_frame.paste(resize_to(imgs["5","coreg"].convert("RGBA"), col2_width, row2_height), (col1_width,row1_height))

        frames.append(new_frame)

    duration = imgs["1","moving"].info.get("duration", 100)

    out_path = OUT / f"{key}.gif"
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
        disposal=2
    )
    print(f"Saved animated GIF: {out_path.name}")
