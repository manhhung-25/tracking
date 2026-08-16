import argparse
from pathlib import Path
from types import SimpleNamespace

import cv2
from ultralytics import YOLO

from track_bus_people import (
    detections_from_result,
    enhance_for_detection,
    nms,
)


def write_yolo_label(label_path, detections, width, height):
    lines = []
    for det in detections:
        x1, y1, x2, y2 = det.box
        cx = ((x1 + x2) * 0.5) / width
        cy = ((y1 + y2) * 0.5) / height
        bw = (x2 - x1) / width
        bh = (y2 - y1) / height
        if bw <= 0 or bh <= 0:
            continue
        lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    label_path.write_text("\n".join(lines), encoding="utf-8")


def make_args(base_args):
    return SimpleNamespace(
        class_id=0,
        low_conf=base_args.low_conf,
        conf=base_args.conf,
        rear_region=base_args.rear_region,
        rear_conf=base_args.rear_conf,
        rear_low_conf=base_args.rear_low_conf,
        rear_min_box_area=base_args.rear_min_box_area,
        rear_confirm_hits=base_args.rear_confirm_hits,
        confirm_hits=base_args.confirm_hits,
        keypoint_conf=base_args.keypoint_conf,
        detector_mode="pose",
        min_box_area=base_args.min_box_area,
        nms=base_args.nms,
        use_bus_roi=True,
        ignore_bottom_ratio=base_args.ignore_bottom_ratio,
        ignore_side_ratio=base_args.ignore_side_ratio,
    )


def export_dataset(args):
    source = Path(args.source)
    out = Path(args.dataset)
    image_dir = out / "images" / "train"
    label_dir = out / "labels" / "train"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {source}")

    pose_model = YOLO(args.pose_model)
    det_args = make_args(args)
    frame_idx = 0
    saved = 0
    model_conf = min(args.low_conf, args.rear_low_conf)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if frame_idx % args.stride != 0:
            continue

        inference_frame = enhance_for_detection(frame) if args.enhance else frame
        result = pose_model.predict(inference_frame, imgsz=args.imgsz, conf=model_conf, iou=args.nms, verbose=False)[0]
        detections = detections_from_result(frame, result, det_args)

        if args.rear_zoom:
            h, _ = frame.shape[:2]
            crop_bottom = max(1, min(h, int(h * args.rear_zoom_bottom)))
            rear_crop = inference_frame[:crop_bottom, :]
            zoomed = cv2.resize(rear_crop, None, fx=args.rear_zoom_scale, fy=args.rear_zoom_scale, interpolation=cv2.INTER_CUBIC)
            rear_result = pose_model.predict(zoomed, imgsz=args.rear_imgsz, conf=model_conf, iou=args.nms, verbose=False)[0]
            detections.extend(detections_from_result(frame, rear_result, det_args, scale=args.rear_zoom_scale))
            detections = nms(detections, args.nms)

        if len(detections) < args.min_labels:
            continue

        stem = f"bus_{frame_idx:06d}"
        cv2.imwrite(str(image_dir / f"{stem}.jpg"), frame)
        height, width = frame.shape[:2]
        write_yolo_label(label_dir / f"{stem}.txt", detections, width, height)
        saved += 1
        if args.max_images and saved >= args.max_images:
            break

    cap.release()
    yaml_path = out / "data.yaml"
    yaml_path.write_text(
        f"path: {out.resolve().as_posix()}\ntrain: images/train\nval: images/train\nnames:\n  0: head_shoulder\n",
        encoding="utf-8",
    )
    print(f"exported images: {saved}")
    print(f"data: {yaml_path}")
    return yaml_path


def train(args, yaml_path):
    model = YOLO(args.base_model)
    model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=args.train_imgsz,
        batch=args.batch,
        device="cpu",
        project="runs_head_train",
        name="bus_head_pseudo",
        exist_ok=True,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Pseudo-train a bus-specific head/shoulder detector.")
    parser.add_argument("--source", default="School Bus - Real Time Live Monitoring (CCTV Camera Systems).mp4")
    parser.add_argument("--dataset", default="bus_head_dataset")
    parser.add_argument("--pose-model", default="yolo11s-pose.pt")
    parser.add_argument("--base-model", default="yolo11n.pt")
    parser.add_argument("--stride", type=int, default=12)
    parser.add_argument("--max-images", type=int, default=180)
    parser.add_argument("--min-labels", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--train-imgsz", type=int, default=960)
    parser.add_argument("--enhance", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rear-zoom", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rear-zoom-bottom", type=float, default=0.64)
    parser.add_argument("--rear-zoom-scale", type=float, default=2.0)
    parser.add_argument("--rear-imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.32)
    parser.add_argument("--low-conf", type=float, default=0.08)
    parser.add_argument("--rear-region", type=float, default=0.58)
    parser.add_argument("--rear-conf", type=float, default=0.20)
    parser.add_argument("--rear-low-conf", type=float, default=0.045)
    parser.add_argument("--rear-min-box-area", type=float, default=45.0)
    parser.add_argument("--rear-confirm-hits", type=int, default=12)
    parser.add_argument("--confirm-hits", type=int, default=4)
    parser.add_argument("--keypoint-conf", type=float, default=0.22)
    parser.add_argument("--min-box-area", type=float, default=120.0)
    parser.add_argument("--nms", type=float, default=0.72)
    parser.add_argument("--ignore-bottom-ratio", type=float, default=0.88)
    parser.add_argument("--ignore-side-ratio", type=float, default=0.025)
    parser.add_argument("--no-train", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    yaml_path = export_dataset(args)
    if not args.no_train:
        train(args, yaml_path)


if __name__ == "__main__":
    main()
