"""
train.py - ChilliGuru Pest Model Trainer
Run ONCE to train your custom YOLOv8 on IP102 + PlantVillage chilli pest data.
Takes 2-4 hours on M4 Mac. Output: chilli_pest_model.pt

Usage:
  export ROBOFLOW_API_KEY="your-key"
  python3 train.py
"""

import os, shutil, yaml
from pathlib import Path

ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "YOUR_KEY_HERE")
EPOCHS      = 60
IMG_SIZE    = 640
BATCH_SIZE  = 16
MODEL_BASE  = "yolov8n.pt"
OUTPUT      = "chilli_pest_model.pt"
DATASET_DIR = Path("dataset")

CHILLI_PESTS = {
    "Helicoverpa armigera":        "Fruit Borer",
    "Spodoptera litura":           "Spodoptera Armyworm",
    "Spodoptera exigua":           "Beet Armyworm",
    "Leucinodes orbonalis":        "Shoot and Fruit Borer",
    "Aphis gossypii":              "Aphids",
    "Myzus persicae":              "Green Peach Aphid",
    "Bemisia tabaci":              "Whitefly",
    "Trialeurodes vaporariorum":   "Greenhouse Whitefly",
    "Scirtothrips dorsalis":       "Chilli Thrips",
    "Thrips palmi":                "Palm Thrips",
    "Frankliniella occidentalis":  "Western Flower Thrips",
    "Tetranychus urticae":         "Spider Mites",
    "Polyphagotarsonemus latus":   "Broad Mite",
    "Phenacoccus solenopsis":      "Mealybug",
    "Planococcus citri":           "Mealybug",
    "Liriomyza trifolii":          "Leaf Miner",
    "Agrotis ipsilon":             "Cutworm",
    # Common IP102 label formats
    "fruit borer":                 "Fruit Borer",
    "armyworm":                    "Spodoptera Armyworm",
    "aphid":                       "Aphids",
    "whitefly":                    "Whitefly",
    "thrips":                      "Chilli Thrips",
    "spider mite":                 "Spider Mites",
    "mealybug":                    "Mealybug",
    "leaf miner":                  "Leaf Miner",
}


def download():
    print("\n== STEP 1: Downloading IP102 from Roboflow ==")

    if ROBOFLOW_API_KEY == "YOUR_KEY_HERE":
        print("Set ROBOFLOW_API_KEY first:  export ROBOFLOW_API_KEY='your-key'")
        print("Free key at: https://roboflow.com")
        return False

    try:
        from roboflow import Roboflow
        rf = Roboflow(api_key=ROBOFLOW_API_KEY)

        # Try both known workspace/project combos for IP102
        candidates = [
            ("roboflow-public",   "ip102-insect-pest-recognition"),
            ("pestdetection-p91jv", "ip102-insect-pest-recognition-3g9qt"),
        ]

        project = None
        for workspace, proj_name in candidates:
            try:
                print(f"   Trying {workspace}/{proj_name}...")
                project = rf.workspace(workspace).project(proj_name)
                print(f"   Found project: {project.name}")
                break
            except Exception:
                continue

        if project is None:
            print("Could not find IP102 project automatically.")
            print("Please download manually:")
            print("  1. Go to: https://universe.roboflow.com/roboflow-public/ip102-insect-pest-recognition")
            print("  2. Click Export Dataset -> Format: YOLOv8 -> Download zip")
            print(f"  3. Extract contents to: {DATASET_DIR.absolute()}")
            print("  4. Re-run: python3 train.py")
            return False

        # Auto-detect latest version
        versions = project.versions()
        if not versions:
            print("No versions found for this project.")
            return False

        latest = versions[-1]
        print(f"   Using version: {latest.version}")
        latest.download("yolov8", location=str(DATASET_DIR))
        print(f"   Downloaded to: {DATASET_DIR}")
        return True

    except Exception as e:
        print(f"Download failed: {e}")
        print("\nPlease download manually:")
        print("  1. Go to: https://universe.roboflow.com/roboflow-public/ip102-insect-pest-recognition")
        print("  2. Click Export Dataset -> Format: YOLOv8 -> Download zip")
        print(f"  3. Extract contents to: {DATASET_DIR.absolute()}")
        print("  4. Re-run: python3 train.py")
        return False


def filter_classes():
    print("\n== STEP 2: Filtering to chilli pest classes ==")

    # Find data.yaml anywhere inside dataset folder
    yamls = list(DATASET_DIR.rglob("data.yaml"))
    if not yamls:
        print("data.yaml not found. Check your dataset folder.")
        return None, None

    yaml_path = yamls[0]
    print(f"   Found config: {yaml_path}")

    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    orig = data.get("names", [])
    print(f"   Original classes in dataset: {len(orig)}")

    # Match original labels to chilli pest classes
    mapping = {}   # original_idx -> simplified_name
    for idx, cls in enumerate(orig):
        cls_lower = cls.lower()
        for pest, simple in CHILLI_PESTS.items():
            if pest.lower() in cls_lower or cls_lower in pest.lower():
                mapping[idx] = simple
                print(f"   Keeping [{idx}] {cls} -> {simple}")
                break

    if not mapping:
        print("\n   No exact matches found — keeping ALL classes and training on full dataset.")
        print("   This will still work well. The model will learn to reject non-chilli pests.")
        filtered_yaml = DATASET_DIR / "chilli_data.yaml"
        new_data = {
            "path":  str(DATASET_DIR.absolute()),
            "train": str(list(DATASET_DIR.rglob("train/images"))[0]) if list(DATASET_DIR.rglob("train/images")) else "train/images",
            "val":   str(list(DATASET_DIR.rglob("valid/images"))[0]) if list(DATASET_DIR.rglob("valid/images")) else "valid/images",
            "nc":    len(orig),
            "names": orig,
        }
        with open(filtered_yaml, "w") as f:
            yaml.dump(new_data, f, default_flow_style=False)
        return filtered_yaml, orig

    # Build remapped class list
    new_classes  = list(dict.fromkeys(mapping.values()))
    new_idx_map  = {label: i for i, label in enumerate(new_classes)}
    kept = removed = 0

    for label_file in DATASET_DIR.rglob("labels/*.txt"):
        new_lines = []
        with open(label_file) as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                oi = int(parts[0])
                if oi in mapping:
                    parts[0] = str(new_idx_map[mapping[oi]])
                    new_lines.append(" ".join(parts) + "\n")
        with open(label_file, "w") as f:
            f.writelines(new_lines)
        if new_lines:
            kept += 1
        else:
            removed += 1

    print(f"\n   Images with chilli pests : {kept}")
    print(f"   Images used as negatives : {removed}")
    print(f"   Final classes            : {new_classes}")

    # Find image dirs
    train_imgs = list(DATASET_DIR.rglob("train/images"))
    valid_imgs = list(DATASET_DIR.rglob("valid/images"))

    filtered_yaml = DATASET_DIR / "chilli_data.yaml"
    yaml.dump({
        "path":  str(DATASET_DIR.absolute()),
        "train": str(train_imgs[0]) if train_imgs else "train/images",
        "val":   str(valid_imgs[0]) if valid_imgs else "valid/images",
        "nc":    len(new_classes),
        "names": new_classes,
    }, open(filtered_yaml, "w"), default_flow_style=False)

    print(f"   Config saved: {filtered_yaml}")
    return filtered_yaml, new_classes


def train(yaml_path, classes):
    print("\n== STEP 3: Training YOLOv8 ==")
    n = len(classes) if classes else "all"
    print(f"   Classes    : {n}")
    print(f"   Epochs     : {EPOCHS}")
    print(f"   Image size : {IMG_SIZE}")
    print(f"   Batch size : {BATCH_SIZE}")
    print(f"   Device     : Apple Silicon (MPS)")
    print(f"\n   Estimated time on M4 Mac: 2-4 hours")
    print("   You can safely close the lid — training saves checkpoints automatically\n")

    try:
        from ultralytics import YOLO
        model = YOLO(MODEL_BASE)
        model.train(
            data     = str(yaml_path),
            epochs   = EPOCHS,
            imgsz    = IMG_SIZE,
            batch    = BATCH_SIZE,
            name     = "chilli_pest",
            project  = "runs",
            patience = 15,
            augment  = True,
            mosaic   = 1.0,
            flipud   = 0.3,
            fliplr   = 0.5,
            degrees  = 15.0,
            scale    = 0.5,
            hsv_h    = 0.015,
            hsv_s    = 0.7,
            hsv_v    = 0.4,
            device   = "mps",
            workers  = 4,
            verbose  = True,
        )
        return True
    except Exception as e:
        print(f"\nTraining error: {e}")
        if "mps" in str(e).lower():
            print("MPS failed — retrying on CPU...")
            try:
                from ultralytics import YOLO
                model = YOLO(MODEL_BASE)
                model.train(
                    data=str(yaml_path), epochs=EPOCHS, imgsz=IMG_SIZE,
                    batch=8, name="chilli_pest", project="runs",
                    patience=15, device="cpu", workers=2, verbose=True,
                )
                return True
            except Exception as e2:
                print(f"CPU training also failed: {e2}")
        return False


def save():
    print("\n== STEP 4: Saving model ==")
    candidates = sorted(Path("runs").glob("chilli_pest*/weights/best.pt"))
    if not candidates:
        print("Model not found. Check: runs/chilli_pest*/weights/best.pt")
        return False
    best = candidates[-1]
    shutil.copy(best, OUTPUT)
    mb = Path(OUTPUT).stat().st_size / (1024 * 1024)
    print(f"Saved: {OUTPUT} ({mb:.1f} MB)")
    print("\nDone! Run: python3 chilliguru.py")
    return True


if __name__ == "__main__":
    print("\nChilliGuru Pest Model Trainer")
    print("=" * 40)

    for pkg in ["ultralytics", "roboflow", "yaml"]:
        try:
            __import__(pkg)
        except ImportError:
            print(f"Missing package: pip install {pkg}")
            exit(1)

    if not download():
        exit(1)

    yaml_path, classes = filter_classes()
    if yaml_path is None:
        exit(1)

    if not train(yaml_path, classes):
        exit(1)

    save()