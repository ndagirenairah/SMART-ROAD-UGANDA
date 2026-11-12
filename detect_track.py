"""Detect, track, and classify vehicle movement in a road video.

The script uses Ultralytics YOLO's ByteTrack integration. It never fabricates
 detections: every CSV row and annotation comes from a frame processed by the
 model.

Example:
    python detect_track.py --input road.mp4 --direction north
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict, deque
from pathlib import Path
from typing import Deque, Dict, Iterable, Tuple

import cv2
from ultralytics import YOLO


VEHICLE_CLASSES = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
}
DIRECTION_ANGLES = {"east": 0.0, "south": 90.0, "west": 180.0, "north": 270.0}
Point = Tuple[float, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLO + ByteTrack and classify vehicle travel direction."
    )
    parser.add_argument("--input", type=Path, required=True, help="Input video file.")
    parser.add_argument(
        "--output-video",
        type=Path,
        default=Path("outputs/annotated.mp4"),
        help="Annotated video path.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs/tracks.csv"),
        help="Per-track evidence CSV path.",
    )
    parser.add_argument(
        "--model",
        default="yolo11n.pt",
        help="Ultralytics model weights (downloaded by Ultralytics if absent).",
    )
    parser.add_argument(
        "--direction",
        choices=sorted(DIRECTION_ANGLES),
        required=True,
        help="Permitted travel direction in the image.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=45.0,
        help="Maximum angular difference from permitted direction.",
    )
    parser.add_argument(
        "--min-displacement",
        type=float,
        default=8.0,
        help="Minimum pixel displacement before classifying a track.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the annotated video while processing.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=320,
        help="Inference image size. Lower this on memory-constrained CPUs.",
    )
    return parser.parse_args()


def angular_difference(first: float, second: float) -> float:
    """Return the smallest absolute difference between two angles."""
    return abs((first - second + 180.0) % 360.0 - 180.0)


def movement_angle(start: Point, end: Point) -> float:
    """Return an image-coordinate angle, where positive y points south."""
    return math.degrees(math.atan2(end[1] - start[1], end[0] - start[0])) % 360.0


def classify_track(
    points: Iterable[Point],
    permitted_angle: float,
    min_displacement: float,
    threshold: float,
) -> Tuple[str, float | None]:
    point_list = list(points)
    if len(point_list) < 2:
        return "insufficient track", None

    start, end = point_list[0], point_list[-1]
    displacement = math.hypot(end[0] - start[0], end[1] - start[1])
    if displacement < min_displacement:
        return "insufficient movement", None

    observed_angle = movement_angle(start, end)
    difference = angular_difference(observed_angle, permitted_angle)
    label = "normal direction" if difference <= threshold else "wrong-way"
    return label, observed_angle


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(f"Input video does not exist: {args.input}")

    ensure_parent(args.output_video)
    ensure_parent(args.output_csv)

    capture = cv2.VideoCapture(str(args.input))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open input video: {args.input}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    writer = cv2.VideoWriter(
        str(args.output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"OpenCV could not create output video: {args.output_video}")

    model = YOLO(args.model)
    tracks: Dict[int, Deque[Point]] = defaultdict(lambda: deque(maxlen=30))
    records: Dict[int, dict] = {}
    permitted_angle = DIRECTION_ANGLES[args.direction]
    frame_number = 0

    try:
        with args.output_csv.open("w", newline="", encoding="utf-8") as csv_file:
            fieldnames = [
                "track_id",
                "class",
                "first_frame",
                "last_frame",
                "observed_angle_degrees",
                "permitted_direction",
                "classification",
                "mean_detection_confidence",
            ]
            csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            csv_writer.writeheader()

            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                frame_number += 1
                result = model.track(
                    frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    imgsz=args.imgsz,
                    classes=list(VEHICLE_CLASSES),
                    verbose=False,
                )[0]
                annotated = frame.copy()

                if result.boxes.id is not None:
                    ids = result.boxes.id.int().cpu().tolist()
                    boxes = result.boxes.xyxy.cpu().tolist()
                    classes = result.boxes.cls.int().cpu().tolist()
                    confidences = result.boxes.conf.cpu().tolist()

                    for track_id, box, class_id, confidence in zip(
                        ids, boxes, classes, confidences
                    ):
                        x1, y1, x2, y2 = map(int, box)
                        center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                        tracks[track_id].append(center)
                        classification, observed_angle = classify_track(
                            tracks[track_id],
                            permitted_angle,
                            args.min_displacement,
                            args.threshold,
                        )
                        record = records.setdefault(
                            track_id,
                            {
                                "track_id": track_id,
                                "class": VEHICLE_CLASSES.get(class_id, str(class_id)),
                                "first_frame": frame_number,
                                "last_frame": frame_number,
                                "observed_angle_degrees": observed_angle,
                                "permitted_direction": args.direction,
                                "classification": classification,
                                "mean_detection_confidence": confidence,
                                "_confidence_total": 0.0,
                                "_confidence_count": 0,
                            },
                        )
                        record["last_frame"] = frame_number
                        record["classification"] = classification
                        if observed_angle is not None:
                            record["observed_angle_degrees"] = round(observed_angle, 2)
                        record["_confidence_total"] += confidence
                        record["_confidence_count"] += 1
                        record["mean_detection_confidence"] = round(
                            record["_confidence_total"] / record["_confidence_count"], 4
                        )

                        color = {
                            "normal direction": (50, 190, 110),
                            "wrong-way": (70, 90, 230),
                        }.get(classification, (80, 203, 246))
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                        label = f"#{track_id} {record['class']} | {classification}"
                        cv2.putText(
                            annotated,
                            label,
                            (x1, max(y1 - 8, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            color,
                            2,
                            cv2.LINE_AA,
                        )
                        points = list(tracks[track_id])
                        for start, end in zip(points, points[1:]):
                            cv2.line(
                                annotated,
                                tuple(map(int, start)),
                                tuple(map(int, end)),
                                color,
                                2,
                            )

                cv2.putText(
                    annotated,
                    f"Permitted: {args.direction} | Frame: {frame_number}",
                    (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                writer.write(annotated)
                if args.show:
                    cv2.imshow("SmartRoad detection", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

            for record in records.values():
                record = {
                    key: value
                    for key, value in record.items()
                    if not key.startswith("_")
                }
                csv_writer.writerow(record)
    finally:
        capture.release()
        writer.release()
        cv2.destroyAllWindows()

    print(f"Annotated video: {args.output_video}")
    print(f"Track evidence:  {args.output_csv}")
    print(f"Tracks found:    {len(records)}")


if __name__ == "__main__":
    main()
