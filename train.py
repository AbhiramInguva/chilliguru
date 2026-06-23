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
DATASET_DIR = Path(os.environ.get("DATASET_PATH", "dataset"))

CHILLI_PESTS = {
    # Mappings to "aphids"
    "Aphis gossypii":              "aphids",
    "Myzus persicae":              "aphids",
    "Green Peach Aphid":           "aphids",
    "aphid":                       "aphids",
    
    # Mappings to "whitefly_leaf_damage"
    "Bemisia tabaci":              "whitefly_leaf_damage",
    "Trialeurodes vaporariorum":   "whitefly_leaf_damage",
    "Greenhouse Whitefly":         "whitefly_leaf_damage",
    "whitefly":                    "whitefly_leaf_damage",
    
    # Mappings to "fruit_borer"
    "Helicoverpa armigera":        "fruit_borer",
    "Leucinodes orbonalis":        "fruit_borer",
    "Shoot and Fruit Borer":       "fruit_borer",
    "fruit borer":                 "fruit_borer",
    
    # Mappings to "tobacco_caterpillar"
    "Spodoptera litura":           "tobacco_caterpillar",
    "Spodoptera exigua":           "tobacco_caterpillar",
    "Spodoptera Armyworm":         "tobacco_caterpillar",
    "Beet Armyworm":               "tobacco_caterpillar",
    "armyworm":                    "tobacco_caterpillar",
    
    # Mappings to "yellow_thrips"
    "Scirtothrips dorsalis":       "yellow_thrips",
    "Chilli Thrips":               "yellow_thrips",
    "thrips":                      "yellow_thrips",
    
    # Mappings to "broad_mites"
    "Polyphagotarsonemus latus":   "broad_mites",
    "Broad Mite":                  "broad_mites",
    
    # Mappings to "invasive_black_thrips"
    "Thrips palmi":                "invasive_black_thrips",
    "Palm Thrips":                 "invasive_black_thrips",
    "Frankliniella occidentalis":  "invasive_black_thrips",
    "Western Flower Thrips":       "invasive_black_thrips",
    
    # Mappings to "mealybugs"
    "Phenacoccus solenopsis":      "mealybugs",
    "Planococcus citri":           "mealybugs",
    "Mealybug":                    "mealybugs",
    "mealybug":                    "mealybugs",
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

    NEW_CLASSES = [
        "aphids",
        "whitefly_leaf_damage",
        "fruit_borer",
        "tobacco_caterpillar",
        "yellow_thrips",
        "broad_mites",
        "invasive_black_thrips",
        "mealybugs"
    ]
    new_idx_map = {name: i for i, name in enumerate(NEW_CLASSES)}

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
    print(f"   Final classes            : {NEW_CLASSES}")

    # Find image dirs
    train_imgs = list(DATASET_DIR.rglob("train/images"))
    valid_imgs = list(DATASET_DIR.rglob("valid/images"))

    filtered_yaml = Path("pan_india_pests.yaml").absolute()
    yaml.dump({
        "path":  f"./{DATASET_DIR}",
        "train": str(train_imgs[0].relative_to(DATASET_DIR.absolute())) if train_imgs else "train/images",
        "val":   str(valid_imgs[0].relative_to(DATASET_DIR.absolute())) if valid_imgs else "valid/images",
        "nc":    len(NEW_CLASSES),
        "names": NEW_CLASSES,
    }, open(filtered_yaml, "w"), default_flow_style=False)

    print(f"   Config saved: {filtered_yaml}")
    return filtered_yaml, NEW_CLASSES


def validate_negative_anchors(yaml_path):
    print("\n== STEP 2.5: Validating background negative anchors ==")
    try:
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        
        dataset_path = Path(data.get("path", "."))
        train_relative = data.get("train", "train/images")
        
        train_images_path = Path(train_relative)
        if not train_images_path.is_absolute():
            train_images_path = dataset_path / train_images_path
            
        if not train_images_path.exists():
            print(f"   [WARNING] Training image directory {train_images_path} does not exist.")
            return False
            
        valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}
        image_files = [f for f in train_images_path.glob("**/*") if f.suffix.lower() in valid_exts]
        
        if not image_files:
            print("   [WARNING] No training images found to validate.")
            return False
            
        total_images = len(image_files)
        background_count = 0
        
        for img_path in image_files:
            # Construct corresponding label path in YOLO format
            parts = list(img_path.parts)
            if "images" in parts:
                idx = len(parts) - 1 - parts[::-1].index("images")
                parts[idx] = "labels"
            label_path = Path(*parts).with_suffix(".txt")
            
            if not label_path.exists():
                background_count += 1
            else:
                try:
                    if label_path.stat().st_size == 0:
                        background_count += 1
                    else:
                        with open(label_path, "r") as lf:
                            content = lf.read().strip()
                            if not content:
                                background_count += 1
                except Exception:
                    background_count += 1
                    
        pct_background = (background_count / total_images) * 100
        print(f"   Total training images: {total_images}")
        print(f"   Background-only images: {background_count} ({pct_background:.2f}%)")
        
        if pct_background < 10.0:
            raise ValueError(f"Hardened pipeline verification failed: negative anchor ratio is {pct_background:.2f}% (required: >= 10.0%)")
        
        print("   ✅ Dataset validation successful: background images pool is >= 10.0%")
        return True
    except Exception as e:
        print(f"   ❌ Validation error: {e}")
        raise e


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
        # Patch Albumentations to inject high-intensity HueSaturationValue transform
        try:
            from ultralytics.data.augment import Albumentations
            import albumentations as A
            original_init = Albumentations.__init__
            def patched_init(self, p=1.0, transforms=None):
                if transforms is None:
                    transforms = [
                        A.Blur(p=0.01),
                        A.MedianBlur(p=0.01),
                        A.ToGray(p=0.2),
                        A.CLAHE(p=0.01),
                        A.RandomBrightnessContrast(p=0.0),
                        A.RandomGamma(p=0.0),
                        A.ImageCompression(quality_range=(75, 100), p=0.0),
                        A.HueSaturationValue(hue_shift_limit=180, sat_shift_limit=40, val_shift_limit=30, p=0.8),
                        # Several different pests/diseases (mites, leaf curl virus, etc.)
                        # all produce a curled/distorted leaf silhouette — without this,
                        # the model can latch onto "curled shape" itself as a shortcut
                        # signal instead of the actual class-distinguishing texture/colour
                        # cues. Randomly warping leaf shape on EVERY training image (not
                        # just the curl-causing classes) breaks that shortcut. Only affects
                        # future training runs — detector.py inference is untouched.
                        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.15),
                        A.ElasticTransform(alpha=1, sigma=50, p=0.15),
                    ]
                original_init(self, p=p, transforms=transforms)
            Albumentations.__init__ = patched_init
            print("   [INFO] Injected HueSaturationValue(hue_shift_limit=180, sat_shift_limit=40, val_shift_limit=30, p=0.8), ToGray(p=0.2), GridDistortion(p=0.15), and ElasticTransform(p=0.15) into Albumentations configuration block.")
        except Exception as patch_err:
            print(f"   [WARNING] Failed to patch Albumentations: {patch_err}")

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
            cls      = 2.5,
            box      = 8.5,
            label_smoothing = 0.1,
            close_mosaic = 15,
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
                    cls=2.5, box=8.5, label_smoothing=0.1, close_mosaic=15,
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

    # Post-training export to ONNX with INT8 quantization & simplification
    try:
        from ultralytics import YOLO
        print("   Exporting model to ONNX INT8 format...")
        model = YOLO(OUTPUT)
        
        dest = Path("weights/chilli_pest_model.onnx")
        dest.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Attempt direct export
            exported_path_str = model.export(format="onnx", int8=True, simplify=True)
            exported_path = Path(exported_path_str)
            if exported_path.exists():
                shutil.copy(exported_path, dest)
                print(f"   Saved ONNX model to: {dest} ({dest.stat().st_size / (1024*1024):.2f} MB)")
            else:
                raise FileNotFoundError("Exported path does not exist")
        except Exception as direct_err:
            print(f"   [INFO] Direct ONNX int8 export failed ({direct_err}). Falling back to standard export + ONNX runtime dynamic quantization...")
            exported_path_str = model.export(format="onnx", simplify=True)
            exported_path = Path(exported_path_str)
            
            from onnxruntime.quantization import quantize_dynamic, QuantType
            quantize_dynamic(
                model_input=str(exported_path),
                model_output=str(dest),
                weight_type=QuantType.QUInt8
            )
            print(f"   Saved ONNX model (dynamic quantized) to: {dest} ({dest.stat().st_size / (1024*1024):.2f} MB)")
            
    except Exception as export_err:
        print(f"   [ERROR] Failed to export/quantize model to ONNX: {export_err}")

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

    # Validate negative anchors pool
    validate_negative_anchors(yaml_path)

    if not train(yaml_path, classes):
        exit(1)

    save()