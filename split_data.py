import random, shutil
from pathlib import Path

random.seed(42)

RAW_DIR = Path("raw_data")
OUT_DIR = Path("data")
SPLIT = (0.8, 0.1, 0.1)  # train, val, test

def main():
    if not RAW_DIR.exists():
        raise FileNotFoundError("raw_data not found. Put class folders inside raw_data/")

    for split in ["train", "val", "test"]:
        (OUT_DIR / split).mkdir(parents=True, exist_ok=True)

    classes = [d.name for d in RAW_DIR.iterdir() if d.is_dir()]
    print("Classes found:", classes)

    for cls in classes:
        imgs = [p for p in (RAW_DIR / cls).iterdir() if p.suffix.lower() in [".jpg",".jpeg",".png",".webp"]]
        random.shuffle(imgs)

        n = len(imgs)
        n_train = int(SPLIT[0] * n)
        n_val = int(SPLIT[1] * n)

        splits = {
            "train": imgs[:n_train],
            "val": imgs[n_train:n_train+n_val],
            "test": imgs[n_train+n_val:]
        }

        for split, files in splits.items():
            dest = OUT_DIR / split / cls
            dest.mkdir(parents=True, exist_ok=True)
            for f in files:
                shutil.copy2(f, dest / f.name)

        print(f"{cls}: total={n}, train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")

if __name__ == "__main__":
    main()
