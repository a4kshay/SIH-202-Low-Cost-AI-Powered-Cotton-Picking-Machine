# infer_images_with_scores.py
from ultralytics import YOLO
import os, csv, argparse
from statistics import mean

parser = argparse.ArgumentParser()
parser.add_argument("--weights", default=r"C:\Users\Akshay\Desktop\kapas_picker08\runs\cotton_maturity_exp\weights\best.pt")
parser.add_argument("--source", default="test_images", help="folder with images")
parser.add_argument("--outdir", default="inference_results", help="output dir")
parser.add_argument("--imgsz", type=int, default=640)
parser.add_argument("--conf", type=float, default=0.25)
parser.add_argument("--mature_class", type=int, default=2)
args = parser.parse_args()

os.makedirs(args.outdir, exist_ok=True)
os.makedirs(os.path.join(args.outdir,"images"), exist_ok=True)

print("✅ infer_images_with_scores STARTED")
print("Using weights:", args.weights)
print("Reading images from:", args.source)

model = YOLO(args.weights)

images = []
for f in sorted(os.listdir(args.source)):
    if f.lower().endswith((".jpg",".jpeg",".png")):
        images.append(os.path.join(args.source,f))
if not images:
    print("No images found in", args.source)
    raise SystemExit

summary_rows = []
for p in images:
    print("➡ Processing:", p)
    results = model.predict(source=p, imgsz=args.imgsz, conf=args.conf, verbose=False)
    r = results[0]
    mature_count = 0
    other_count = 0
    mature_confs = []
    # boxes might be empty
    if getattr(r, "boxes", None) is not None and len(r.boxes) > 0:
        for box in r.boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            if cls_id == args.mature_class:
                mature_count += 1
                mature_confs.append(conf)
            else:
                other_count += 1

    total = mature_count + other_count
    mean_conf = mean(mature_confs) if mature_confs else 0.0
    maturity_percent = (len(mature_confs)/max(1,total)) * mean_conf * 100 if total>0 else 0.0

    # save annotated image
    out_img = os.path.join(args.outdir, "images", os.path.basename(p))
    annotated = r.plot()
    import cv2
    cv2.imwrite(out_img, annotated)

    summary_rows.append({
        "image": os.path.basename(p),
        "mature_count": mature_count,
        "other_count": other_count,
        "maturity_percent": f"{maturity_percent:.2f}",
        "mature_conf_mean": f"{mean_conf:.3f}"
    })

# write CSV
csv_path = os.path.join(args.outdir, "summary.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as fh:
    writer = csv.DictWriter(fh, fieldnames=summary_rows[0].keys())
    writer.writeheader()
    writer.writerows(summary_rows)

print("✅ Inference complete.")
print("Outputs saved to:", args.outdir)
print("CSV summary:", csv_path)
