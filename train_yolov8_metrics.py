from ultralytics import YOLO
import argparse
import os
import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data',
        type=str,
        default=r"C:\Users\Akshay\Desktop\kapas_picker08\Cotton Boll Detection.v1i.yolov8\data.yaml",
        help='Path to data.yaml (YOLOv8 format dataset)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='yolov8n.pt',
        help='Base model or .pt file'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=45,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--imgsz',
        type=int,
        default=640,
        help='Image size for training/validation'
    )
    parser.add_argument(
        '--batch',
        type=int,
        default=16,
        help='Batch size'
    )
    parser.add_argument(
        '--name',
        type=str,
        default='cotton_maturity_exp',
        help='Experiment name (run folder name prefix)'
    )
    args = parser.parse_args()

    # --------------------------------------------------
    # 1️⃣ Show dataset configuration
    # --------------------------------------------------
    if not os.path.exists(args.data):
        raise FileNotFoundError(f"data.yaml not found at: {args.data}")

    with open(args.data, 'r') as f:
        data_cfg = yaml.safe_load(f)

    nc = data_cfg.get('nc', None)
    names = data_cfg.get('names', None)

    print("📁 Using dataset config:", args.data)
    print(f"   -> nc    = {nc}")
    print(f"   -> names = {names}")

    # --------------------------------------------------
    # 2️⃣ Load base model (pretrained or previous .pt)
    # --------------------------------------------------
    print(f"\n🔍 Loading model: {args.model}")
    model = YOLO(args.model)

    # --------------------------------------------------
    # 3️⃣ Train the model
    # --------------------------------------------------
    print("\n🚀 Starting training...")
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,   # run name, auto-increments if exists
        project="runs",   # results in runs/detect/<nameX>
        verbose=True
    )

    # Get save directory of this run
    save_dir = str(model.trainer.save_dir)   # e.g. runs/detect/cotton_maturity_exp2
    weights_dir = os.path.join(save_dir, "weights")
    best_weights = os.path.join(weights_dir, "best.pt")
    last_weights = os.path.join(weights_dir, "last.pt")

    print("\n✅ Training finished.")
    print("   Save directory:", save_dir)
    print("   Best weights  :", best_weights)
    print("   Last weights  :", last_weights)

    if not os.path.exists(best_weights):
        print("⚠️ best.pt not found, falling back to last.pt")
        best_weights = last_weights

    # --------------------------------------------------
    # 4️⃣ Validate best model and print metrics
    # --------------------------------------------------
    print("\n📊 Running validation on best model...")
    best_model = YOLO(best_weights)

    metrics = best_model.val(
        data=args.data,
        imgsz=args.imgsz,
        split='val',   # uses the "valid" split from data.yaml
        verbose=False
    )

    box = metrics.box  # Ultralytics metrics object

    print("\n================= VALIDATION METRICS =================")
    try:
        print(f"mAP@0.5:         {box.map50:.4f}")
        print(f"mAP@0.5:0.95:    {box.map:.4f}")
        print(f"Mean Precision:  {box.mp:.4f}")
        print(f"Mean Recall:     {box.mr:.4f}")
    except AttributeError:
        print("⚠️ Could not read aggregate metrics from metrics.box")

    # Per-class AP
    if names and hasattr(box, "maps"):
        print("\nPer-class mAP@0.5:")
        for i, ap in enumerate(box.maps):
            if i < len(names):
                print(f"  class {i} ({names[i]}): {ap:.4f}")
            else:
                print(f"  class {i}: {ap:.4f}")
    else:
        print("\nPer-class metrics not available.")

    print("======================================================\n")
    print("🎯 Done. You can open this folder for plots, PR curves, and predictions:")
    print(f"   {save_dir}")


if __name__ == "__main__":
    main()
        