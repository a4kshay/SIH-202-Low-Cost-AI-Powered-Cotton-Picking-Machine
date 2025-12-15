print("✅ infer_cotton_maturity.py STARTED")

from ultralytics import YOLO
import os

# Class mapping from your dataset (from data.yaml):
# 0 = Defected boll - Don't pick
# 1 = Flower - Don't pick
# 2 = Fully opened boll - Ready to pick  ✅ MATURE
# 3 = Partially opened - Not ready to pick

CLASS_NAMES = [
    "Defected boll - Don't pick",
    "Flower - Don't pick",
    "Fully opened boll - Ready to pick",
    "Partially opened - Not ready to pick",
]

MATURE_CLASS_ID = 2  # class 2 is mature boll (ready to pick)


def main():
    print("🔧 main() started")

    # 🔧 Paths (change only if you moved files)
    weights_path = r"C:\Users\Akshay\Desktop\kapas_picker08\runs\cotton_maturity_exp\weights\best.pt"
    source_dir = r"test_images"  # folder with test photos (relative to this script)
    output_dir = os.path.join("inference_results", "images")

    print(f"📁 Current working dir: {os.getcwd()}")
    print(f"🧠 Using weights      : {weights_path}")
    print(f"🖼 Reading images from: {source_dir}")

    # ✅ Check if best.pt exists
    if not os.path.exists(weights_path):
        print(f"❌ ERROR: best.pt not found at: {weights_path}")
        return

    # ✅ Check source folder
    if not os.path.isdir(source_dir):
        print(f"❌ ERROR: Source folder not found: {source_dir}")
        print("   👉 Create this folder and put some cotton images inside.")
        return

    os.makedirs(output_dir, exist_ok=True)

    # 🔍 Load YOLO model
    print("🔍 Loading YOLO model...")
    model = YOLO(weights_path)

    # 🖼 Collect all image files
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    image_files = [
        f for f in os.listdir(source_dir)
        if f.lower().endswith(exts)
    ]

    if not image_files:
        print("⚠️ No images found in test_images folder (jpg/png/bmp).")
        return

    print(f"📸 Found {len(image_files)} image(s). Starting inference...\n")

    from cv2 import imwrite  # import here to avoid error if cv2 missing

    for filename in image_files:
        img_path = os.path.join(source_dir, filename)
        print(f"➡ Processing: {img_path}")

        # Run YOLO inference
        results = model.predict(
            source=img_path,
            imgsz=640,
            conf=0.25,
            save=False,
            verbose=False,
        )

        r = results[0]
        mature_count = 0
        not_mature_count = 0

        # Count detections by class
        if r.boxes is not None and len(r.boxes) > 0:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                if cls_id == MATURE_CLASS_ID:
                    mature_count += 1
                else:
                    not_mature_count += 1

        status = "MATURE PRESENT ✅" if mature_count > 0 else "NO MATURE BOLL ❌"

        print(f"   Mature bolls (class 2) : {mature_count}")
        print(f"   Other / not-ready      : {not_mature_count}")
        print(f"   ⇒ Status               : {status}")

        # Save annotated image with boxes
        annotated = r.plot()  # numpy array with drawings
        out_path = os.path.join(output_dir, filename)
        imwrite(out_path, annotated)
        print(f"   💾 Saved annotated image to: {os.path.abspath(out_path)}\n")

    print("\n✅ Inference complete.")
    print("📂 All outputs saved in:", os.path.abspath(output_dir))


if __name__ == "__main__":
    print("📢 __name__ == '__main__' → calling main()")
    main()
