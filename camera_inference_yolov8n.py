# camera_inference_fast_yolov8n.py
import cv2, time, os
from ultralytics import YOLO
import torch
from statistics import mean

WEIGHTS_TORCH = r"C:\Users\Akshay\Desktop\kapas_picker08\runs\cotton_maturity_exp\weights\best.torchscript.pt"
WEIGHTS_PT = r"C:\Users\Akshay\Desktop\kapas_picker08\runs\cotton_maturity_exp\weights\best.pt"
CAM_INDEX = 0
IMG_SIZE = 320  # smaller for speed
CONF = 0.25
MATURE_CLASS = 2
SMOOTH_N = 3

weights = WEIGHTS_TORCH if os.path.exists(WEIGHTS_TORCH) else WEIGHTS_PT
print("Loading model:", weights)
model = YOLO(weights)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW if os.name == "nt" else 0)
if not cap.isOpened():
    raise SystemExit("Cannot open camera")

recent = []
while True:
    t0 = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(source=frame, imgsz=IMG_SIZE, conf=CONF, device=device, verbose=False)
    infer_ms = (time.time() - t0) * 1000
    r = results[0]

    mature_count = 0
    confs = []
    if getattr(r, "boxes", None) is not None:
        for b in r.boxes:
            cls = int(b.cls[0].item())
            conf = float(b.conf[0].item())
            if cls == MATURE_CLASS:
                mature_count += 1
                confs.append(conf)

    top = max(confs) if confs else 0.0
    recent.append(top)
    if len(recent) > SMOOTH_N:
        recent.pop(0)
    smooth = mean(recent) if recent else 0.0

    annotated = r.plot()
    overlay = f"Infer {infer_ms:.0f}ms  Mature:{mature_count}  top:{top:.2f} smooth:{smooth:.2f}"
    cv2.rectangle(annotated, (0,0),(450,40),(0,0,0),-1)
    cv2.putText(annotated, overlay, (8,24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255),1)
    cv2.imshow("fast infer", annotated)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
