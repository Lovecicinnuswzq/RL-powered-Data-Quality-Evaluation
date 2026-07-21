import argparse
import json
import os
import torch
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(
        description="Train YOLO sub-process predictor and evaluate performance."
    )
    parser.add_argument("--model_path", type=str, required=True, help="Path to initial YOLO model.")
    parser.add_argument("--yaml_path", type=str, required=True, help="Path to data YAML file.")
    parser.add_argument("--val_dir", type=str, required=True, help="Path to validation images.")
    parser.add_argument("--output_json", type=str, required=True, help="Path to save predictions JSON.")
    args = parser.parse_args()

    model = YOLO(args.model_path)
    model.train(
        data=args.yaml_path,
        epochs=30,
        batch=4,
        exist_ok=True,
        save=False,
    )

    val_results = model.val(data=args.yaml_path, split="val", verbose=False)

    metrics = {
        "map50": float(val_results.results_dict.get("metrics/mAP50(B)", val_results.box.map50)),
        "map5095": float(val_results.results_dict.get("metrics/mAP50-95(B)", val_results.box.map)),
        "map75": float(val_results.results_dict.get("metrics/mAP75(B)", val_results.box.map75)),
    }

    metrics_json_path = os.path.join(os.path.dirname(args.output_json), "yolo_val_metrics.json")
    os.makedirs(os.path.dirname(metrics_json_path), exist_ok=True)
    with open(metrics_json_path, "w") as f:
        json.dump(metrics, f)

    val_preds = {}
    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
    val_img_files = sorted(
        [f for f in os.listdir(args.val_dir) if f.lower().endswith(valid_extensions)]
    )

    for fname in val_img_files:
        img_path = os.path.join(args.val_dir, fname)
        pred = model.predict(img_path, save=False, verbose=False)[0]

        boxes = pred.boxes.xywh.cpu().numpy().tolist() if (
                    pred.boxes is not None and pred.boxes.xywh is not None) else []
        confs = pred.boxes.conf.cpu().numpy().tolist() if (
                    pred.boxes is not None and pred.boxes.conf is not None) else []
        val_preds[fname] = {"boxes": boxes, "confs": confs}

    with open(args.output_json, "w") as f:
        json.dump(val_preds, f)


if __name__ == "__main__":
    main()