from ultralytics import YOLO
import argparse
import os
import json

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", required=True)
parser.add_argument("--yaml_path", required=True)
parser.add_argument("--val_dir", required=True)
parser.add_argument("--output_json", required=True)
args = parser.parse_args()

model = YOLO(args.model_path)
model.train(data=args.yaml_path, epochs=30, imgsz=640, batch=8,
            mosaic=0.0, mixup=0.0, copy_paste=0.0, exist_ok=True, save=False, workers=4)

val_preds = {}
val_img_files = sorted([f for f in os.listdir(args.val_dir) if f.endswith(".jpg")])
for fname in val_img_files:
    img_path = os.path.join(args.val_dir, fname)
    pred = model.predict(img_path, save=False, verbose=False)[0]
    boxes = pred.boxes.xywh.cpu().numpy().tolist() if pred.boxes.xywh is not None else []
    confs = pred.boxes.conf.cpu().numpy().tolist() if pred.boxes.conf is not None else []
    val_preds[fname] = {"boxes": boxes, "confs": confs}

with open(args.output_json, "w") as f:
    json.dump(val_preds, f)
