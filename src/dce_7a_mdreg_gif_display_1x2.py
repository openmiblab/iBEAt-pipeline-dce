from pathlib import Path
from PIL import Image
import re

DIR = Path("build/dce_7_mdreg/Sheffield/Patients/_GIFS")
OUT = Path("build/dce_7_mdreg/Sheffield/Patients/_moco_1x2")
OUT.mkdir(exist_ok=True, parents=True)

# Regex that supports BOTH filename formats
iter_re = re.compile(
    r"(?:"
    r"MDREG_(\d+)_(\d+)_baseline_(1|5)_(coreg|moving)"        # pattern A
    r"|"
    r"MDREG_(coreg|moving)_(\d+)_(\d+)_(1|5)_"                # pattern B
    r")\.gif$",
    re.IGNORECASE
)

# ------------------------------------------
# 1. All GIFs from the directory
# ------------------------------------------
files = {}

for f in DIR.glob("*.gif"):
    m = iter_re.search(f.name)
    if not m:
        print("No match:", f.name)
        continue

    # Pattern A: MDREG_<study>_<subject>_baseline_<baseline>_<type>.gif
    if m.group(1) is not None:
        study   = m.group(1)
        subject = m.group(2)
        baseline = m.group(3)
        gtype   = m.group(4)

    # Pattern B: MDREG_<type>_<study>_<subject>_<baseline>_.gif
    else:
        gtype   = m.group(5)
        study   = m.group(6)
        subject = m.group(7)
        baseline = m.group(8)

    key = f"{study}_{subject}"

    files.setdefault(key, {})[(baseline, gtype)] = f


# ------------------------------------------
# 2. Build moving+coreg 1×2 GIFs
# ------------------------------------------
for key, baselines in files.items():
    for baseline in ["1", "5"]:
        required = [(baseline, "moving"), (baseline, "coreg")]

        if any(r not in baselines for r in required):
            print(f"Skipping {key} baseline {baseline}: missing GIFs")
            continue

        imgs = {r: Image.open(baselines[r]) for r in required}

        width = max(imgs[(baseline, "moving")].width,
                    imgs[(baseline, "coreg")].width)
        height = max(imgs[(baseline, "moving")].height,
                     imgs[(baseline, "coreg")].height)

        frames = []
        n_frames = min(getattr(imgs[r], "n_frames", 1) for r in required)

        for i in range(n_frames):
            for r in required:
                imgs[r].seek(i)

            new_frame = Image.new("RGBA", (width*2, height))
            new_frame.paste(
                imgs[(baseline, "moving")].copy().resize((width, height)), (0, 0)
            )
            new_frame.paste(
                imgs[(baseline, "coreg")].copy().resize((width, height)), (width, 0)
            )
            frames.append(new_frame)

        out_path = OUT / f"{key}_{baseline}.gif"
        frames[0].save(
            out_path,
            save_all=True,
            append_images=frames[1:],
            duration=imgs[(baseline, "moving")].info.get("duration", 100),
            loop=0
        )

        print(f"Saved {out_path.name}")
