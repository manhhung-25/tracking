import argparse
import csv
import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO


COCO_HEAD_SHOULDER = [0, 1, 2, 3, 4, 5, 6]
COCO_HEAD = [0, 1, 2, 3, 4]


def clip_box(box, width, height):
    x1, y1, x2, y2 = box
    x1 = max(0.0, min(float(width - 1), x1))
    y1 = max(0.0, min(float(height - 1), y1))
    x2 = max(0.0, min(float(width - 1), x2))
    y2 = max(0.0, min(float(height - 1), y2))
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def box_area(box):
    return max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))


def iou(a, b):
    xx1 = max(float(a[0]), float(b[0]))
    yy1 = max(float(a[1]), float(b[1]))
    xx2 = min(float(a[2]), float(b[2]))
    yy2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, xx2 - xx1) * max(0.0, yy2 - yy1)
    denom = box_area(a) + box_area(b) - inter
    return 0.0 if denom <= 0 else inter / denom


def center(box):
    return np.array([(box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5], dtype=np.float32)


def expand_box(box, width, height, sx=1.45, sy=1.75):
    cx, cy = center(box)
    bw = max(1.0, float(box[2] - box[0])) * sx
    bh = max(1.0, float(box[3] - box[1])) * sy
    return clip_box([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], width, height)


def inside_bus_roi(box, width, height, args):
    cx, cy = center(box)
    if cy >= height * args.ignore_bottom_ratio:
        return False
    if cx <= width * args.ignore_side_ratio or cx >= width * (1.0 - args.ignore_side_ratio):
        return False

    y_ratio = cy / max(1.0, height)
    left_ratio = 0.02 + max(0.0, 0.32 - y_ratio) * 0.65
    right_ratio = 0.98 - max(0.0, 0.32 - y_ratio) * 0.65
    return width * left_ratio <= cx <= width * right_ratio


def nms(detections, threshold):
    if not detections:
        return []
    detections = sorted(detections, key=lambda det: det.score, reverse=True)
    kept = []
    while detections:
        current = detections.pop(0)
        kept.append(current)
        detections = [det for det in detections if iou(current.box, det.box) < threshold]
    return kept


def hsv_hist(frame, box):
    x1, y1, x2, y2 = box.astype(int)
    crop = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
    if crop.size == 0:
        return np.zeros(48, dtype=np.float32)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 3], [0, 180, 0, 256]).flatten()
    norm = np.linalg.norm(hist)
    return (hist / norm).astype(np.float32) if norm > 0 else hist.astype(np.float32)


def enhance_for_detection(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    enhanced = cv2.cvtColor(cv2.merge((l_channel, a_channel, b_channel)), cv2.COLOR_LAB2BGR)
    blurred = cv2.GaussianBlur(enhanced, (0, 0), 1.1)
    return cv2.addWeighted(enhanced, 1.35, blurred, -0.35, 0)


def cosine_distance(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom <= 1e-6:
        return 1.0
    return float(1.0 - np.dot(a, b) / denom)


def make_head_box_from_pose(person_box, keypoints, confs, width, height, min_kpt_conf):
    visible_head = [i for i in COCO_HEAD if confs[i] >= min_kpt_conf]
    visible_anchor = [i for i in COCO_HEAD_SHOULDER if confs[i] >= min_kpt_conf]

    if visible_head:
        pts = keypoints[visible_head]
        x1, y1 = pts.min(axis=0)
        x2, y2 = pts.max(axis=0)
        span = max(float(x2 - x1), float(y2 - y1), 18.0)
        pad_x = span * 0.55
        pad_top = span * 0.65
        pad_bottom = span * 0.95
        return clip_box([x1 - pad_x, y1 - pad_top, x2 + pad_x, y2 + pad_bottom], width, height)

    if visible_anchor:
        pts = keypoints[visible_anchor]
        x1, y1 = pts.min(axis=0)
        x2, y2 = pts.max(axis=0)
        shoulder_width = max(float(x2 - x1), 28.0)
        anchor_y = float(y1)
        return clip_box(
            [x1 - shoulder_width * 0.10, anchor_y - shoulder_width * 0.95,
             x2 + shoulder_width * 0.10, anchor_y + shoulder_width * 0.38],
            width,
            height,
        )

    x1, y1, x2, y2 = person_box
    person_h = y2 - y1
    person_w = x2 - x1
    return clip_box(
        [x1 + person_w * 0.22, y1, x2 - person_w * 0.22, y1 + person_h * 0.30],
        width,
        height,
    )


@dataclass
class Detection:
    box: np.ndarray
    score: float
    embedding: np.ndarray
    edge: str = ""
    startable: bool = False
    required_hits: int = 4


@dataclass
class Track:
    track_id: int
    box: np.ndarray
    embedding: np.ndarray
    hits: int = 1
    required_hits: int = 4
    age: int = 0
    missed: int = 0
    visible: bool = True
    last_seen_frame: int = 0
    start_edge: str = ""
    last_edge: str = ""
    counted_in: bool = False
    last_detected_box: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float32))
    klt_points: np.ndarray | None = None
    history: list = field(default_factory=list)
    state: np.ndarray = field(default_factory=lambda: np.zeros(8, dtype=np.float32))

    def __post_init__(self):
        cx, cy = center(self.box)
        w = self.box[2] - self.box[0]
        h = self.box[3] - self.box[1]
        self.state = np.array([cx, cy, w, h, 0, 0, 0, 0], dtype=np.float32)
        self.last_detected_box = self.box.copy()

    def predict(self):
        self.state[:4] += self.state[4:]
        cx, cy, w, h = self.state[:4]
        self.box = np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=np.float32)
        self.age += 1
        self.missed += 1
        self.history.append(center(self.box))
        self.history = self.history[-40:]

    def optical_flow_predict(self, prev_gray, gray, frame_size):
        if prev_gray is None or gray is None or self.klt_points is None or len(self.klt_points) < 4:
            self.predict()
            return

        next_points, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray,
            gray,
            self.klt_points,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
        )
        if next_points is None or status is None:
            self.predict()
            return

        good_prev = self.klt_points[status.flatten() == 1]
        good_next = next_points[status.flatten() == 1]
        if len(good_next) < 4:
            self.klt_points = None
            self.predict()
            return

        shifts = good_next.reshape(-1, 2) - good_prev.reshape(-1, 2)
        dx, dy = np.median(shifts, axis=0)
        if abs(float(dx)) > 45 or abs(float(dy)) > 45:
            self.klt_points = None
            self.predict()
            return

        width, height = frame_size
        shifted = self.box + np.array([dx, dy, dx, dy], dtype=np.float32)
        self.box = clip_box(shifted, width, height)
        cx, cy = center(self.box)
        self.state[4:] = 0.55 * self.state[4:] + 0.45 * np.array([dx, dy, 0, 0], dtype=np.float32)
        self.state[0] = cx
        self.state[1] = cy
        self.age += 1
        self.missed += 1
        self.klt_points = good_next.reshape(-1, 1, 2).astype(np.float32)
        self.history.append(center(self.box))
        self.history = self.history[-40:]

    def init_klt(self, gray):
        x1, y1, x2, y2 = self.box.astype(int)
        h, w = gray.shape[:2]
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(x1 + 2, min(w, x2))
        y2 = max(y1 + 2, min(h, y2))
        roi = gray[y1:y2, x1:x2]
        if roi.size == 0:
            self.klt_points = None
            return
        points = cv2.goodFeaturesToTrack(
            roi,
            maxCorners=24,
            qualityLevel=0.01,
            minDistance=3,
            blockSize=3,
        )
        if points is None:
            self.klt_points = None
            return
        points[:, 0, 0] += x1
        points[:, 0, 1] += y1
        self.klt_points = points.astype(np.float32)

    def update(self, detection, frame_index, gray=None, alpha=0.80):
        new_cx, new_cy = center(detection.box)
        new_w = detection.box[2] - detection.box[0]
        new_h = detection.box[3] - detection.box[1]
        measurement = np.array([new_cx, new_cy, new_w, new_h], dtype=np.float32)
        velocity = measurement - self.state[:4]
        self.state[:4] = measurement
        self.state[4:] = 0.65 * self.state[4:] + 0.35 * velocity
        self.box = detection.box.copy()
        self.last_detected_box = detection.box.copy()
        self.embedding = alpha * self.embedding + (1.0 - alpha) * detection.embedding
        norm = np.linalg.norm(self.embedding)
        if norm > 0:
            self.embedding = self.embedding / norm
        self.hits += 1
        self.missed = 0
        self.visible = True
        self.last_seen_frame = frame_index
        self.last_edge = detection.edge
        if gray is not None:
            self.init_klt(gray)
        self.history.append(center(self.box))
        self.history = self.history[-40:]


@dataclass
class LostIdentity:
    track_id: int
    box: np.ndarray
    embedding: np.ndarray
    last_seen_frame: int
    last_edge: str


class HeadShoulderTracker:
    def __init__(
        self,
        fps,
        frame_size,
        high_thresh=0.35,
        low_thresh=0.12,
        match_thresh=0.74,
        track_buffer_seconds=3.0,
        reid_buffer_seconds=18.0,
        reid_weight=0.32,
        edge_margin=0.08,
        duplicate_iou=0.72,
        count_warmup_seconds=1.0,
        confirm_hits=4,
        start_suppression=1.35,
    ):
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.match_thresh = match_thresh
        self.max_missed = max(1, int(round(track_buffer_seconds * fps)))
        self.max_lost_age = max(1, int(round(reid_buffer_seconds * fps)))
        self.reid_weight = reid_weight
        self.width, self.height = frame_size
        self.edge_margin = edge_margin
        self.duplicate_iou = duplicate_iou
        self.count_start_frame = int(round(count_warmup_seconds * fps))
        self.confirm_hits = confirm_hits
        self.start_suppression = start_suppression
        self.tracks = []
        self.lost = []
        self.next_id = 1
        self.in_count = 0
        self.out_count = 0

    def update(self, detections, frame_index, gray=None, prev_gray=None):
        for det in detections:
            det.edge = self._edge_for_box(det.box)

        for track in self.tracks:
            track.optical_flow_predict(prev_gray, gray, (self.width, self.height))

        high = [det for det in detections if det.startable]
        low = [det for det in detections if self.low_thresh <= det.score and not det.startable]

        unmatched_tracks = list(range(len(self.tracks)))
        unmatched_high = list(range(len(high)))
        matches = self._match(unmatched_tracks, high)
        for track_idx, det_idx in matches:
            self.tracks[track_idx].update(high[det_idx], frame_index, gray=gray)
            self._maybe_count_in(self.tracks[track_idx], frame_index)
        unmatched_tracks = [i for i in unmatched_tracks if all(i != m[0] for m in matches)]
        unmatched_high = [i for i in unmatched_high if all(i != m[1] for m in matches)]

        low_matches = self._match(unmatched_tracks, low, allow_weak=True)
        for track_idx, det_idx in low_matches:
            self.tracks[track_idx].update(low[det_idx], frame_index, gray=gray, alpha=0.90)
            self._maybe_count_in(self.tracks[track_idx], frame_index)
        unmatched_tracks = [i for i in unmatched_tracks if all(i != m[0] for m in low_matches)]

        for det_idx in unmatched_high:
            if not self._near_existing_identity(high[det_idx]):
                self._start_or_reid_track(high[det_idx], frame_index, gray)

        kept = []
        for track in self.tracks:
            if track.missed <= self.max_missed:
                kept.append(track)
            else:
                self._retire_track(track, frame_index)
        self.tracks = kept
        self._prune_lost(frame_index)
        self._suppress_duplicate_tracks()
        return [track for track in self.tracks if track.missed == 0 and track.hits >= track.required_hits]

    def _start_or_reid_track(self, detection, frame_index, gray=None):
        lost_idx = self._match_lost_identity(detection, frame_index)
        if lost_idx is not None:
            identity = self.lost.pop(lost_idx)
            track = Track(identity.track_id, detection.box.copy(), detection.embedding.copy())
            track.required_hits = detection.required_hits
            track.hits = detection.required_hits
            track.last_seen_frame = frame_index
            track.start_edge = detection.edge
            track.last_edge = detection.edge
            if gray is not None:
                track.init_klt(gray)
            self.tracks.append(track)
            self._maybe_count_in(track, frame_index)
            return

        track = Track(0, detection.box.copy(), detection.embedding.copy())
        track.required_hits = detection.required_hits
        track.last_seen_frame = frame_index
        track.start_edge = detection.edge
        track.last_edge = detection.edge
        if gray is not None:
            track.init_klt(gray)
        self.tracks.append(track)

    def _near_existing_identity(self, detection):
        det_center = center(detection.box)
        det_diag = max(1.0, math.hypot(detection.box[2] - detection.box[0], detection.box[3] - detection.box[1]))
        for track in self.tracks:
            if iou(track.box, detection.box) >= 0.05 or iou(track.last_detected_box, detection.box) >= 0.05:
                return True
            track_diag = max(1.0, math.hypot(track.box[2] - track.box[0], track.box[3] - track.box[1]))
            dist = np.linalg.norm(det_center - center(track.box))
            if dist <= max(18.0, min(det_diag, track_diag) * self.start_suppression):
                return True
        for identity in self.lost:
            if iou(identity.box, detection.box) >= 0.03:
                return False
            dist = np.linalg.norm(det_center - center(identity.box))
            if dist <= max(22.0, det_diag * 1.8):
                return False
        return False

    def _retire_track(self, track, frame_index):
        if track.hits < track.required_hits or track.track_id <= 0:
            return
        edge = self._exit_edge_for_box(track.last_detected_box)
        if edge and frame_index > self.count_start_frame:
            self.out_count += 1
        self.lost.append(LostIdentity(
            track_id=track.track_id,
            box=track.last_detected_box.copy(),
            embedding=track.embedding.copy(),
            last_seen_frame=track.last_seen_frame or frame_index,
            last_edge=edge or track.last_edge,
        ))

    def _maybe_count_in(self, track, frame_index):
        if track.hits >= track.required_hits and track.track_id <= 0:
            track.track_id = self.next_id
            self.next_id += 1
        if track.counted_in or track.hits < track.required_hits:
            return
        edge = track.start_edge or track.last_edge
        if edge and frame_index > self.count_start_frame:
            self.in_count += 1
            track.counted_in = True

    def _match(self, track_indices, detections, allow_weak=False):
        if not track_indices or not detections:
            return []

        cost = np.ones((len(track_indices), len(detections)), dtype=np.float32) * 1e6
        for row, track_idx in enumerate(track_indices):
            track = self.tracks[track_idx]
            for col, det in enumerate(detections):
                ov = iou(track.box, det.box)
                c_dist = np.linalg.norm(center(track.box) - center(det.box))
                diag = max(1.0, math.hypot(track.box[2] - track.box[0], track.box[3] - track.box[1]))
                motion_cost = min(1.0, c_dist / (diag * 3.0))
                appearance_cost = cosine_distance(track.embedding, det.embedding)
                total = (1.0 - ov) * (1.0 - self.reid_weight) + appearance_cost * self.reid_weight
                total = 0.70 * total + 0.30 * motion_cost
                if ov >= 0.03 or motion_cost < (0.82 if allow_weak else 0.62):
                    cost[row, col] = total

        rows, cols = linear_sum_assignment(cost)
        matches = []
        limit = self.match_thresh + (0.10 if allow_weak else 0.0)
        for row, col in zip(rows, cols):
            if cost[row, col] <= limit:
                matches.append((track_indices[row], col))
        return matches

    def _match_lost_identity(self, detection, frame_index):
        best_idx = None
        best_cost = 1e6
        for idx, identity in enumerate(self.lost):
            age = frame_index - identity.last_seen_frame
            if age > self.max_lost_age:
                continue
            app = cosine_distance(identity.embedding, detection.embedding)
            c_dist = np.linalg.norm(center(identity.box) - center(detection.box))
            scene_diag = math.hypot(self.width, self.height)
            motion = min(1.0, c_dist / max(1.0, scene_diag * 0.65))
            edge_bonus = -0.08 if identity.last_edge and detection.edge else 0.0
            cost = 0.45 * app + 0.55 * motion + edge_bonus
            if cost < best_cost:
                best_cost = cost
                best_idx = idx
        return best_idx if best_cost <= 0.50 else None

    def _edge_for_box(self, box):
        cx, cy = center(box)
        margin_x = self.width * self.edge_margin
        margin_y = self.height * self.edge_margin
        edges = []
        if cx <= margin_x:
            edges.append("left")
        if cx >= self.width - margin_x:
            edges.append("right")
        if cy <= margin_y:
            edges.append("top")
        if cy >= self.height - margin_y:
            edges.append("bottom")
        return "+".join(edges)

    def _exit_edge_for_box(self, box):
        cx, cy = center(box)
        hard_x = self.width * 0.025
        hard_y = self.height * 0.025
        edges = []
        if cx <= hard_x or box[2] <= 1:
            edges.append("left")
        if cx >= self.width - hard_x or box[0] >= self.width - 2:
            edges.append("right")
        if cy <= hard_y or box[3] <= 1:
            edges.append("top")
        if cy >= self.height - hard_y or box[1] >= self.height - 2:
            edges.append("bottom")
        return "+".join(edges)

    def _prune_lost(self, frame_index):
        self.lost = [
            identity for identity in self.lost
            if frame_index - identity.last_seen_frame <= self.max_lost_age
        ]

    def _suppress_duplicate_tracks(self):
        if len(self.tracks) < 2:
            return
        remove = set()
        for i in range(len(self.tracks)):
            if i in remove:
                continue
            for j in range(i + 1, len(self.tracks)):
                if j in remove:
                    continue
                ti = self.tracks[i]
                tj = self.tracks[j]
                same_place = iou(ti.box, tj.box) >= self.duplicate_iou
                same_app = cosine_distance(ti.embedding, tj.embedding) <= 0.16
                if same_place or (same_app and np.linalg.norm(center(ti.box) - center(tj.box)) < 18):
                    loser = j if ti.hits >= tj.hits else i
                    remove.add(loser)
        self.tracks = [track for idx, track in enumerate(self.tracks) if idx not in remove]


def detections_from_result(frame, result, args, offset=(0.0, 0.0), scale=1.0):
    height, width = frame.shape[:2]
    detections = []
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return detections

    keypoints_xy = None
    keypoints_conf = None
    if result.keypoints is not None and result.keypoints.xy is not None:
        keypoints_xy = result.keypoints.xy.cpu().numpy()
        keypoints_xy = keypoints_xy / scale
        keypoints_xy[:, :, 0] += offset[0]
        keypoints_xy[:, :, 1] += offset[1]
        keypoints_conf = result.keypoints.conf.cpu().numpy() if result.keypoints.conf is not None else None

    person_boxes = boxes.xyxy.cpu().numpy()
    person_boxes = person_boxes / scale
    person_boxes[:, [0, 2]] += offset[0]
    person_boxes[:, [1, 3]] += offset[1]
    scores = boxes.conf.cpu().numpy()
    classes = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else np.zeros(len(person_boxes), dtype=int)

    for idx, person_box in enumerate(person_boxes):
        if args.class_id >= 0 and classes[idx] != args.class_id:
            continue
        score = float(scores[idx])
        person_center_y = float((person_box[1] + person_box[3]) * 0.5)
        is_rear = person_center_y <= height * args.rear_region
        min_conf = args.rear_low_conf if is_rear else args.low_conf
        min_area = args.rear_min_box_area if is_rear else args.min_box_area
        if score < min_conf:
            continue
        if args.detector_mode == "head":
            box = clip_box(person_box, width, height)
        elif keypoints_xy is not None and idx < len(keypoints_xy):
            confs = keypoints_conf[idx] if keypoints_conf is not None else np.ones(17, dtype=np.float32)
            box = make_head_box_from_pose(person_box, keypoints_xy[idx], confs, width, height, args.keypoint_conf)
        else:
            box = make_head_box_from_pose(person_box, np.zeros((17, 2), dtype=np.float32), np.zeros(17), width, height, args.keypoint_conf)
        if box_area(box) < min_area:
            continue
        if args.use_bus_roi and not inside_bus_roi(box, width, height, args):
            continue
        startable = score >= (args.rear_conf if is_rear else args.conf)
        detections.append(Detection(
            box=box,
            score=score,
            embedding=hsv_hist(frame, expand_box(box, width, height)),
            startable=startable,
            required_hits=args.rear_confirm_hits if is_rear else args.confirm_hits,
        ))
    return nms(detections, args.nms)


def color_for_id(track_id):
    rng = np.random.default_rng(track_id * 9973)
    return tuple(int(v) for v in rng.integers(50, 240, size=3))


def draw_tracks(frame, tracks, frame_index, total_seen, in_count, out_count):
    cv2.rectangle(frame, (8, 8), (310, 38), (15, 15, 15), -1)
    cv2.putText(frame, f"F{frame_index} A{len(tracks)} IDs{total_seen} IN {in_count} OUT {out_count}",
                (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    for track in tracks:
        x1, y1, x2, y2 = track.box.astype(int)
        color = color_for_id(track.track_id)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
        label = f"ID {track.track_id}"
        cv2.putText(frame, label, (x1 + 2, max(10, y1 - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, color, 1, cv2.LINE_AA)
        pts = [tuple(p.astype(int)) for p in track.history[-20:]]
        for p1, p2 in zip(pts, pts[1:]):
            cv2.line(frame, p1, p2, color, 1)


def show_frame(window_name, frame, scale):
    if scale != 1.0:
        frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    cv2.imshow(window_name, frame)
    key = cv2.waitKey(1) & 0xFF
    return key not in (27, ord("q"))


def parse_args():
    parser = argparse.ArgumentParser(description="Head/shoulder multi-object tracking for occluded bus CCTV video.")
    parser.add_argument("--source", default="School Bus - Real Time Live Monitoring (CCTV Camera Systems).mp4")
    parser.add_argument("--output", default="tracked_bus_people.mp4")
    parser.add_argument("--csv", default="tracked_bus_people.csv")
    parser.add_argument("--show", action="store_true",
                        help="Show the tracking result on screen while processing. Press q or Esc to stop.")
    parser.add_argument("--display-scale", type=float, default=1.0,
                        help="Resize display window only. Output video keeps original size.")
    parser.add_argument("--model", default="yolo11s-pose.pt",
                        help="Use a custom fine-tuned head model here if you have one; pose model is the default.")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--enhance", action=argparse.BooleanOptionalAction, default=True,
                        help="Enhance contrast/sharpness before detection; tracking is still drawn on the original frame.")
    parser.add_argument("--detector-mode", choices=["pose", "head"], default="pose",
                        help="pose derives head/shoulder boxes; head uses model boxes directly for a fine-tuned head detector.")
    parser.add_argument("--class-id", type=int, default=0, help="Class to keep. Use -1 to keep all classes.")
    parser.add_argument("--conf", type=float, default=0.32)
    parser.add_argument("--low-conf", type=float, default=0.08)
    parser.add_argument("--rear-region", type=float, default=0.58,
                        help="Top portion of the image treated as far/rear seats.")
    parser.add_argument("--rear-conf", type=float, default=0.20,
                        help="Start threshold for blurry/small people in the rear region.")
    parser.add_argument("--rear-low-conf", type=float, default=0.045,
                        help="Low threshold for associating blurry/small rear detections.")
    parser.add_argument("--rear-min-box-area", type=float, default=45.0)
    parser.add_argument("--rear-confirm-hits", type=int, default=12,
                        help="Frames required before blurry/small rear detections become visible IDs.")
    parser.add_argument("--rear-zoom", action=argparse.BooleanOptionalAction, default=True,
                        help="Run an additional zoomed detector pass on the rear/top part of the bus.")
    parser.add_argument("--rear-zoom-bottom", type=float, default=0.64,
                        help="Bottom Y ratio of the rear crop used by --rear-zoom.")
    parser.add_argument("--rear-zoom-scale", type=float, default=2.0)
    parser.add_argument("--rear-imgsz", type=int, default=960)
    parser.add_argument("--use-bus-roi", action=argparse.BooleanOptionalAction, default=True,
                        help="Ignore edge/overlay areas outside the useful bus interior.")
    parser.add_argument("--ignore-bottom-ratio", type=float, default=0.88)
    parser.add_argument("--ignore-side-ratio", type=float, default=0.025)
    parser.add_argument("--keypoint-conf", type=float, default=0.22)
    parser.add_argument("--nms", type=float, default=0.72)
    parser.add_argument("--match-thresh", type=float, default=0.72)
    parser.add_argument("--track-buffer", type=float, default=8.0,
                        help="Seconds to keep IDs alive while detections are missing.")
    parser.add_argument("--reid-buffer", type=float, default=18.0,
                        help="Seconds to keep retired edge tracks available for ReID.")
    parser.add_argument("--reid-weight", type=float, default=0.25)
    parser.add_argument("--edge-margin", type=float, default=0.08,
                        help="Frame margin ratio used to count IN/OUT at image borders.")
    parser.add_argument("--duplicate-iou", type=float, default=0.68)
    parser.add_argument("--count-warmup", type=float, default=1.0,
                        help="Seconds ignored before counting border IN events from initial occupants.")
    parser.add_argument("--confirm-hits", type=int, default=4,
                        help="Frames required before a new ID is displayed and counted.")
    parser.add_argument("--start-suppression", type=float, default=1.35,
                        help="Suppress new IDs near existing tracks; higher means fewer duplicate IDs.")
    parser.add_argument("--min-box-area", type=float, default=120.0)
    parser.add_argument("--max-frames", type=int, default=0, help="Debug only. 0 processes the full video.")
    return parser.parse_args()


def main():
    args = parse_args()
    source = Path(args.source)
    if not source.exists():
        raise FileNotFoundError(f"Input video not found: {source}")

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    writer = cv2.VideoWriter(
        args.output,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create output video: {args.output}")

    model = YOLO(args.model)
    tracker = HeadShoulderTracker(
        fps=fps,
        frame_size=(width, height),
        high_thresh=args.conf,
        low_thresh=min(args.low_conf, args.rear_low_conf),
        match_thresh=args.match_thresh,
        track_buffer_seconds=args.track_buffer,
        reid_buffer_seconds=args.reid_buffer,
        reid_weight=args.reid_weight,
        edge_margin=args.edge_margin,
        duplicate_iou=args.duplicate_iou,
        count_warmup_seconds=args.count_warmup,
        confirm_hits=args.confirm_hits,
        start_suppression=args.start_suppression,
    )
    seen_ids = set()
    frame_index = 0
    prev_gray = None
    window_name = "Bus people tracking"
    if args.show:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    with open(args.csv, "w", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["frame", "track_id", "x1", "y1", "x2", "y2", "cx", "cy", "edge", "in_count", "out_count"])

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_index += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            inference_frame = enhance_for_detection(frame) if args.enhance else frame
            model_conf = min(args.low_conf, args.rear_low_conf)
            result = model.predict(inference_frame, imgsz=args.imgsz, conf=model_conf, iou=args.nms, verbose=False)[0]
            detections = detections_from_result(frame, result, args)
            if args.rear_zoom:
                crop_bottom = max(1, min(height, int(height * args.rear_zoom_bottom)))
                rear_crop = inference_frame[:crop_bottom, :]
                zoomed = cv2.resize(
                    rear_crop,
                    None,
                    fx=args.rear_zoom_scale,
                    fy=args.rear_zoom_scale,
                    interpolation=cv2.INTER_CUBIC,
                )
                rear_result = model.predict(
                    zoomed,
                    imgsz=args.rear_imgsz,
                    conf=model_conf,
                    iou=args.nms,
                    verbose=False,
                )[0]
                detections.extend(detections_from_result(
                    frame,
                    rear_result,
                    args,
                    offset=(0.0, 0.0),
                    scale=args.rear_zoom_scale,
                ))
                detections = nms(detections, args.nms)
            active_tracks = tracker.update(detections, frame_index, gray=gray, prev_gray=prev_gray)
            prev_gray = gray
            for track in active_tracks:
                seen_ids.add(track.track_id)
                cx, cy = center(track.box)
                csv_writer.writerow([
                    frame_index,
                    track.track_id,
                    round(float(track.box[0]), 2),
                    round(float(track.box[1]), 2),
                    round(float(track.box[2]), 2),
                    round(float(track.box[3]), 2),
                    round(float(cx), 2),
                    round(float(cy), 2),
                    track.last_edge,
                    tracker.in_count,
                    tracker.out_count,
                ])

            draw_tracks(frame, active_tracks, frame_index, len(seen_ids), tracker.in_count, tracker.out_count)
            writer.write(frame)
            if args.show and not show_frame(window_name, frame, args.display_scale):
                print("stopped by user")
                break

            if frame_index % 100 == 0:
                print(f"processed {frame_index}/{total_frames or '?'} frames, seen IDs={len(seen_ids)}")
            if args.max_frames and frame_index >= args.max_frames:
                break

    cap.release()
    writer.release()
    if args.show:
        cv2.destroyWindow(window_name)
    print(f"done: {args.output}")
    print(f"csv: {args.csv}")
    print(f"unique IDs: {len(seen_ids)}")
    print(f"in: {tracker.in_count}")
    print(f"out: {tracker.out_count}")


if __name__ == "__main__":
    main()
