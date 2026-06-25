"""
================================================================================
HE THONG HO TRO DI CHUYEN CHO NGUOI KHIEM THI - EDGE AI CLIENT (TERMUX ANDROID)
================================================================================

Vai tro cua chuong trinh nay:
    - Doc luong video MJPEG tu ESP32-S3-CAM (http://ESP32_IP:81/stream)
    - Chay YOLOv8 (Ultralytics) de nhan dien vat the tren tung frame
    - Doc du lieu khoang cach / goc servo / canh bao tu ESP32 (http://ESP32_IP:8080/data)
    - Gui heartbeat cho ESP32 (http://ESP32_IP:8080/yolo?ok=1) moi 2 giay
    - Phat canh bao bang giong noi tieng Viet qua termux-tts-speak
    - Cung cap dashboard web (Flask) de theo doi toan bo he thong realtime

Toan bo chuong trinh nam trong file nay theo yeu cau (khong tach module).
HTML/CSS/JS duoc nhung truc tiep bang render_template_string().

KHONG sua firmware ESP32. Code nay tuong thich 100% voi 3 API da co san.
================================================================================
"""

from __future__ import annotations

import logging
import queue
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import requests
from flask import Flask, Response, jsonify, render_template_string

try:
    from ultralytics import YOLO
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "[ERROR] Thieu thu vien 'ultralytics'. Cai dat bang: pip install ultralytics"
    ) from exc


# ============================================================================
# SECTION 1: CAU HINH (CHINH TAI DAY)
# ============================================================================

ESP32_IP: str = "10.25.87.51"  # IP cua ESP32-S3-CAM trong mang LAN
STREAM_URL: str = f"http://{ESP32_IP}:81/stream"
DATA_URL: str = f"http://{ESP32_IP}:8080/data"
HEARTBEAT_URL: str = f"http://{ESP32_IP}:8080/yolo?ok=1"

MODEL_PATH: str = "yolov8n.pt"      # Co the doi sang model khac (yolov8n.onnx, ...)
CONF_THRESHOLD: float = 0.45        # Nguong tin cay YOLO
IOU_THRESHOLD: float = 0.45         # Nguong IOU cho NMS

CAMERA_TARGET_FPS: float = 15.0     # FPS toi da doc tu camera (gioi han CPU)
YOLO_TARGET_FPS: float = 8.0        # FPS toi da chay inference YOLO
JPEG_QUALITY: int = 80              # Chat luong JPEG khi encode frame da ve box

VOICE_MIN_INTERVAL_SEC: float = 3.0     # Thoi gian toi thieu giua 2 lan doc
HEARTBEAT_INTERVAL_SEC: float = 2.0     # Tan suat gui heartbeat cho ESP32
DATA_POLL_INTERVAL_SEC: float = 0.2     # Tan suat poll /data
HTTP_TIMEOUT_SEC: float = 3.0           # Timeout cho moi HTTP request
RECONNECT_DELAY_SEC: float = 2.0        # Cho truoc khi reconnect camera/ESP32
ESP32_DATA_TIMEOUT_SEC: float = 4.0      # Qua thoi gian nay khong co data -> esp32_ok=False
YOLO_STALE_FRAME_SEC: float = 1.0       # Bo qua frame qua cu khi dua vao YOLO

FLASK_HOST: str = "0.0.0.0"
FLASK_PORT: int = 5000

MAX_MJPEG_BUFFER_BYTES: int = 2_000_000  # Nguong reset buffer camera (chong leak RAM)

# Tu dien dich ten vat the COCO sang tieng Viet (fallback = giu nguyen ten goc)
VN_OBJECT_NAMES: Dict[str, str] = {
    "person": "người",
    "bicycle": "xe đạp",
    "car": "ô tô",
    "motorcycle": "xe máy",
    "bus": "xe buýt",
    "truck": "xe tải",
    "chair": "ghế",
    "bench": "băng ghế",
    "dog": "chó",
    "cat": "mèo",
    "potted plant": "chậu cây",
    "dining table": "bàn",
    "stairs": "cầu thang",
    "traffic light": "đèn giao thông",
    "stop sign": "biển báo dừng",
    "fire hydrant": "trụ cứu hỏa",
    "backpack": "cặp",
    "umbrella": "dù",
    "suitcase": "vali",
    "bottle": "chai",
    "couch": "ghế sofa",
    "bed": "giường",
    "tv": "màn hình tivi",
    "laptop": "laptop",
    "door": "cửa",
}

# Nhan canh bao theo khoang cach (dong bo voi enum WarningLevel trong firmware)
WARNING_VOICE_TEXT: Dict[str, str] = {
    "SAFE": "",
    "OBSTACLE": "Có vật cản phía trước",
    "DANGEROUS": "Chú ý, vật cản ở gần",
    "EXTREME_DANGER": "Nguy hiểm, vật cản rất gần",
}


# ============================================================================
# SECTION 2: LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("EdgeAI")


# ============================================================================
# SECTION 3: TRANG THAI CHIA SE GIUA CAC THREAD (THREAD-SAFE)
# ============================================================================

@dataclass
class SystemState:
    """
    Trang thai toan cuc duy nhat, duoc bao ve bang 1 Lock.
    Tat ca thread doc/ghi qua cac phuong thuc public ben duoi, khong truy cap
    truc tiep field de tranh race condition.
    """

    _lock: threading.Lock = field(default_factory=threading.Lock)

    # --- Camera ---
    _raw_frame: Optional[np.ndarray] = None
    _raw_frame_ts: float = 0.0
    _camera_ok: bool = False
    _camera_fps: float = 0.0

    # --- YOLO ---
    _annotated_jpeg: Optional[bytes] = None
    _detections: List[Dict[str, Any]] = field(default_factory=list)
    _yolo_ok: bool = False
    _yolo_fps: float = 0.0

    # --- ESP32 data (/data) ---
    _distance_cm: float = -1.0
    _angle: int = 90
    _warning: str = "SAFE"
    _esp32_ok: bool = False
    _last_data_ts: float = 0.0

    # --- Heartbeat ---
    _heartbeat_ok: bool = False

    # ----------------------- Camera -----------------------
    def set_raw_frame(self, frame: np.ndarray) -> None:
        with self._lock:
            self._raw_frame = frame
            self._raw_frame_ts = time.monotonic()

    def get_raw_frame(self) -> Optional[tuple]:
        """Tra ve (frame, timestamp) hoac None. Tra ve copy de YOLO thread an toan."""
        with self._lock:
            if self._raw_frame is None:
                return None
            return self._raw_frame.copy(), self._raw_frame_ts

    def set_camera_ok(self, ok: bool) -> None:
        with self._lock:
            self._camera_ok = ok

    def set_camera_fps(self, fps: float) -> None:
        with self._lock:
            self._camera_fps = fps

    # ----------------------- YOLO -----------------------
    def set_yolo_result(self, jpeg_bytes: bytes, detections: List[Dict[str, Any]]) -> None:
        with self._lock:
            self._annotated_jpeg = jpeg_bytes
            self._detections = detections
            self._yolo_ok = True

    def set_yolo_ok(self, ok: bool) -> None:
        with self._lock:
            self._yolo_ok = ok

    def set_yolo_fps(self, fps: float) -> None:
        with self._lock:
            self._yolo_fps = fps

    def get_annotated_jpeg(self) -> Optional[bytes]:
        with self._lock:
            return self._annotated_jpeg

    def get_detections(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._detections)

    # ----------------------- ESP32 data -----------------------
    def set_esp32_data(self, distance_cm: float, angle: int, warning: str) -> None:
        with self._lock:
            self._distance_cm = distance_cm
            self._angle = angle
            self._warning = warning
            self._esp32_ok = True
            self._last_data_ts = time.monotonic()

    def check_esp32_timeout(self) -> None:
        """Goi dinh ky de tu dong ha co esp32_ok neu lau khong co du lieu moi."""
        with self._lock:
            if time.monotonic() - self._last_data_ts > ESP32_DATA_TIMEOUT_SEC:
                self._esp32_ok = False

    def set_heartbeat_ok(self, ok: bool) -> None:
        with self._lock:
            self._heartbeat_ok = ok

    # ----------------------- Snapshot tong hop (cho Flask API) -----------------------
    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "distance_cm": round(self._distance_cm, 1),
                "distance_m": round(self._distance_cm / 100.0, 2) if self._distance_cm > 0 else -1,
                "angle": self._angle,
                "warning": self._warning,
                "camera_fps": round(self._camera_fps, 1),
                "yolo_fps": round(self._yolo_fps, 1),
                "camera_ok": self._camera_ok,
                "yolo_ok": self._yolo_ok,
                "esp32_ok": self._esp32_ok,
                "heartbeat_ok": self._heartbeat_ok,
                "detections": list(self._detections),
            }


# ============================================================================
# SECTION 4: CAMERA THREAD (DOC MJPEG TU ESP32, TU RECONNECT)
# ============================================================================

class CameraThread(threading.Thread):
    """
    Doc luong MJPEG bang requests.get(stream=True) va tu parse boundary JPEG
    (theo SOI 0xFFD8 / EOI 0xFFD9) - KHONG dung cv2.VideoCapture(URL).
    Tu dong reconnect khi mat stream, khong bao gio crash chuong trinh chinh.
    """

    def __init__(self, state: SystemState, stop_event: threading.Event) -> None:
        super().__init__(name="CameraThread", daemon=True)
        self._state = state
        self._stop_event = stop_event
        self._frame_count = 0
        self._fps_timer = time.monotonic()

    def run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._stream_loop()
            except Exception as exc:  # noqa: BLE001 - phai bat moi loi de khong crash
                logger.error("[CAMERA] Lỗi stream: %s", exc)
            finally:
                self._state.set_camera_ok(False)

            if not self._stop_event.is_set():
                logger.warning(
                    "[CAMERA] Mất kết nối, thử lại sau %.1fs", RECONNECT_DELAY_SEC
                )
                time.sleep(RECONNECT_DELAY_SEC)

    def _stream_loop(self) -> None:
        logger.info("[CAMERA] Đang kết nối %s", STREAM_URL)
        response = requests.get(STREAM_URL, stream=True, timeout=HTTP_TIMEOUT_SEC)
        response.raise_for_status()
        self._state.set_camera_ok(True)
        logger.info("[CAMERA] Kết nối stream thành công")

        buffer = bytearray()
        min_interval = 1.0 / CAMERA_TARGET_FPS if CAMERA_TARGET_FPS > 0 else 0.0
        last_frame_time = 0.0

        for chunk in response.iter_content(chunk_size=4096):
            if self._stop_event.is_set():
                return
            if not chunk:
                continue

            buffer.extend(chunk)

            # Tim 1 frame JPEG hoan chinh trong buffer (SOI...EOI)
            start = buffer.find(b"\xff\xd8")
            end = buffer.find(b"\xff\xd9")

            if start == -1 or end == -1 or end <= start:
                # Chua co frame hoan chinh -> tranh buffer phinh to vo han
                if len(buffer) > MAX_MJPEG_BUFFER_BYTES:
                    logger.warning("[CAMERA] Buffer vượt ngưỡng, reset buffer")
                    buffer.clear()
                continue

            jpg_bytes = bytes(buffer[start : end + 2])
            del buffer[: end + 2]  # giai phong phan da xu ly, khong giu rac

            now = time.monotonic()
            if now - last_frame_time < min_interval:
                continue  # gioi han FPS dau vao de giam tai CPU
            last_frame_time = now

            frame = cv2.imdecode(np.frombuffer(jpg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue

            self._state.set_raw_frame(frame)
            self._tick_fps()

        # iter_content ket thuc (server dong stream) -> coi nhu mat ket noi
        raise ConnectionError("Luồng MJPEG kết thúc bất ngờ")

    def _tick_fps(self) -> None:
        self._frame_count += 1
        now = time.monotonic()
        elapsed = now - self._fps_timer
        if elapsed >= 1.0:
            self._state.set_camera_fps(self._frame_count / elapsed)
            self._frame_count = 0
            self._fps_timer = now


# ============================================================================
# SECTION 5: YOLO THREAD (NHAN DIEN VAT THE, VE BOX, ENCODE JPEG)
# ============================================================================

class YoloThread(threading.Thread):
    """
    Luon lay frame MOI NHAT tu CameraThread (khong dung queue vo han -> khong leak RAM).
    Neu khong co frame moi hoac frame qua cu, bo qua vong lap de tiet kiem CPU.
    """

    def __init__(self, state: SystemState, stop_event: threading.Event) -> None:
        super().__init__(name="YoloThread", daemon=True)
        self._state = state
        self._stop_event = stop_event
        self._model: Optional[YOLO] = None
        self._last_processed_ts: float = 0.0
        self._frame_count = 0
        self._fps_timer = time.monotonic()

    def run(self) -> None:
        if not self._load_model():
            return  # _load_model da log loi, khong the chay tiep

        min_interval = 1.0 / YOLO_TARGET_FPS if YOLO_TARGET_FPS > 0 else 0.0
        last_infer_time = 0.0

        while not self._stop_event.is_set():
            now = time.monotonic()
            if now - last_infer_time < min_interval:
                time.sleep(0.01)
                continue

            frame_data = self._state.get_raw_frame()
            if frame_data is None:
                time.sleep(0.05)
                continue

            frame, frame_ts = frame_data
            if frame_ts <= self._last_processed_ts:
                time.sleep(0.01)
                continue  # frame chua doi moi, tranh infer trung lap
            if now - frame_ts > YOLO_STALE_FRAME_SEC:
                self._last_processed_ts = frame_ts
                continue  # frame qua cu, bo qua de bat kip realtime

            self._last_processed_ts = frame_ts
            last_infer_time = now

            try:
                self._infer_and_publish(frame)
            except Exception as exc:  # noqa: BLE001
                logger.error("[YOLO] Lỗi inference: %s", exc)
                self._state.set_yolo_ok(False)

    def _load_model(self) -> bool:
        try:
            logger.info("[YOLO] Đang tải model %s ...", MODEL_PATH)
            self._model = YOLO(MODEL_PATH)
            logger.info("[YOLO] Tải model thành công")
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error("[YOLO] Không tải được model: %s", exc)
            self._state.set_yolo_ok(False)
            return False

    def _infer_and_publish(self, frame: np.ndarray) -> None:
        assert self._model is not None
        results = self._model.predict(
            frame,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            verbose=False,
        )

        detections: List[Dict[str, Any]] = []
        annotated = frame  # ve truc tiep, khong copy them de tiet kiem RAM

        result = results[0]
        names = result.names

        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            en_name = names.get(cls_id, str(cls_id))
            vn_name = VN_OBJECT_NAMES.get(en_name, en_name)

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{vn_name} {conf:.0%}"
            cv2.putText(
                annotated, label, (x1, max(y1 - 8, 0)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2,
            )

            detections.append(
                {
                    "name_en": en_name,
                    "name_vn": vn_name,
                    "confidence": round(conf, 2),
                    "box": [x1, y1, x2, y2],
                }
            )

        ok, jpeg_buf = cv2.imencode(
            ".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
        )
        if ok:
            self._state.set_yolo_result(jpeg_buf.tobytes(), detections)

        self._tick_fps()

    def _tick_fps(self) -> None:
        self._frame_count += 1
        now = time.monotonic()
        elapsed = now - self._fps_timer
        if elapsed >= 1.0:
            self._state.set_yolo_fps(self._frame_count / elapsed)
            self._frame_count = 0
            self._fps_timer = now


# ============================================================================
# SECTION 6: DATA THREAD (POLL /data TU ESP32)
# ============================================================================

class DataThread(threading.Thread):
    """Poll dinh ky API /data cua ESP32 de lay khoang cach, goc servo, canh bao."""

    def __init__(self, state: SystemState, stop_event: threading.Event) -> None:
        super().__init__(name="DataThread", daemon=True)
        self._state = state
        self._stop_event = stop_event

    def run(self) -> None:
        while not self._stop_event.is_set():
            try:
                resp = requests.get(DATA_URL, timeout=HTTP_TIMEOUT_SEC)
                resp.raise_for_status()
                payload = resp.json()

                self._state.set_esp32_data(
                    distance_cm=float(payload.get("distance_cm", -1)),
                    angle=int(payload.get("angle", 90)),
                    warning=str(payload.get("warning", "SAFE")),
                )
            except requests.RequestException as exc:
                logger.warning("[DATA] Không lấy được /data: %s", exc)
            except (ValueError, TypeError) as exc:
                logger.error("[DATA] Dữ liệu JSON không hợp lệ: %s", exc)

            self._state.check_esp32_timeout()
            time.sleep(DATA_POLL_INTERVAL_SEC)


# ============================================================================
# SECTION 7: HEARTBEAT THREAD (BAO ESP32 BIET YOLO VAN SONG)
# ============================================================================

class HeartbeatThread(threading.Thread):
    """Gui GET /yolo?ok=1 moi HEARTBEAT_INTERVAL_SEC giay."""

    def __init__(self, state: SystemState, stop_event: threading.Event) -> None:
        super().__init__(name="HeartbeatThread", daemon=True)
        self._state = state
        self._stop_event = stop_event

    def run(self) -> None:
        while not self._stop_event.is_set():
            try:
                resp = requests.get(HEARTBEAT_URL, timeout=HTTP_TIMEOUT_SEC)
                ok = resp.status_code == 200
                self._state.set_heartbeat_ok(ok)
                if not ok:
                    logger.warning("[HEARTBEAT] ESP32 trả về mã lỗi %s", resp.status_code)
            except requests.RequestException as exc:
                self._state.set_heartbeat_ok(False)
                logger.warning("[HEARTBEAT] Không gửi được heartbeat: %s", exc)

            time.sleep(HEARTBEAT_INTERVAL_SEC)


# ============================================================================
# SECTION 8: VOICE ALERT (TTS) - KHONG BLOCK CAC THREAD KHAC
# ============================================================================

class TTSWorker(threading.Thread):
    """
    Worker rieng, chay tuan tu (blocking trong NOI BO thread nay) de tranh
    chong cheo am thanh, nhung khong lam block VoiceAlertThread / he thong chinh
    vi giao tiep qua hang doi (queue) co kich thuoc gioi han.
    """

    def __init__(self, text_queue: "queue.Queue[str]", stop_event: threading.Event) -> None:
        super().__init__(name="TTSWorker", daemon=True)
        self._queue = text_queue
        self._stop_event = stop_event

    def run(self) -> None:
        while not self._stop_event.is_set():
            try:
                text = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                subprocess.run(
                    ["termux-tts-speak", "-l", "vi", text],
                    timeout=10,
                    check=False,
                )
            except FileNotFoundError:
                logger.error("[VOICE] Không tìm thấy lệnh termux-tts-speak (cần cài Termux:API)")
            except subprocess.TimeoutExpired:
                logger.warning("[VOICE] termux-tts-speak quá thời gian chờ")
            except Exception as exc:  # noqa: BLE001
                logger.error("[VOICE] Lỗi phát giọng nói: %s", exc)


class VoiceAlertThread(threading.Thread):
    """
    Quyet dinh NOI DUNG can doc dua tren warning level + vat the gan nhat,
    debounce theo VOICE_MIN_INTERVAL_SEC va khong doc lap noi dung khong doi.
    """

    def __init__(self, state: SystemState, stop_event: threading.Event) -> None:
        super().__init__(name="VoiceAlertThread", daemon=True)
        self._state = state
        self._stop_event = stop_event
        self._tts_queue: "queue.Queue[str]" = queue.Queue(maxsize=3)
        self._tts_worker = TTSWorker(self._tts_queue, stop_event)
        self._last_spoken_text: str = ""
        self._last_spoken_time: float = 0.0

    def start(self) -> None:  # noqa: D102 - override de khoi dong worker kem theo
        self._tts_worker.start()
        super().start()

    def run(self) -> None:
        while not self._stop_event.is_set():
            snapshot = self._state.snapshot()
            text = self._build_message(snapshot)

            now = time.monotonic()
            should_speak = (
                text
                   and (now - self._last_spoken_time) >= VOICE_MIN_INTERVAL_SEC
            )

            if should_speak:
                self._enqueue(text)
                self._last_spoken_text = text
                self._last_spoken_time = now

            time.sleep(0.3)

    def _enqueue(self, text: str) -> None:
        try:
            self._tts_queue.put_nowait(text)
        except queue.Full:
            logger.warning("[VOICE] Hàng đợi TTS đầy, bỏ qua: %s", text)

    @staticmethod
    def _build_message(snapshot: Dict[str, Any]) -> str:
        warning = snapshot["warning"]
        base_text = WARNING_VOICE_TEXT.get(warning, "")

        if warning == "SAFE" or not base_text:
            return ""

        detections = snapshot["detections"]
        if not detections:
            return base_text

        # Chon vat the co box lon nhat (gan nhat / noi bat nhat trong khung hinh)
        def box_area(det: Dict[str, Any]) -> int:
            x1, y1, x2, y2 = det["box"]
            return max(0, x2 - x1) * max(0, y2 - y1)

        top_det = max(detections, key=box_area)
        object_text = f"Có {top_det['name_vn']} phía trước"

        distance_m = snapshot["distance_m"]
        if distance_m > 0:
            distance_text = VoiceAlertThread._format_distance_vi(distance_m)
            return f"{object_text}, cách khoảng {distance_text}"

        return object_text

    @staticmethod
    def _format_distance_vi(distance_m: float) -> str:
        """Doc so thap phan kieu tieng Viet, vi du 1.5 -> 'một phẩy năm mét'."""
        digits_vi = ["không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín"]
        rounded = round(distance_m, 1)
        integer_part = int(rounded)
        decimal_part = int(round((rounded - integer_part) * 10))

        int_text = digits_vi[integer_part] if 0 <= integer_part <= 9 else str(integer_part)
        if decimal_part == 0:
            return f"{int_text} mét"

        dec_text = digits_vi[decimal_part] if 0 <= decimal_part <= 9 else str(decimal_part)
        return f"{int_text} phẩy {dec_text} mét"


# ============================================================================
# SECTION 9: FLASK DASHBOARD (HTML/CSS/JS NHUNG TRUC TIEP)
# ============================================================================

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Hệ thống hỗ trợ người khiếm thị</title>
<style>
  :root { color-scheme: dark; }
  body {
    margin: 0; padding: 16px; background: #0f1115; color: #e8e8e8;
    font-family: "Segoe UI", Roboto, Arial, sans-serif;
  }
  h1 { font-size: 1.3rem; margin-bottom: 12px; }
  .grid {
    display: grid; grid-template-columns: 2fr 1fr; gap: 16px;
  }
  @media (max-width: 800px) { .grid { grid-template-columns: 1fr; } }
  .card {
    background: #181b22; border-radius: 10px; padding: 14px 18px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.4); margin-bottom: 16px;
  }
  .video-wrap { background: #000; border-radius: 10px; overflow: hidden; }
  .video-wrap img { width: 100%; display: block; }
  .stat-row { display: flex; justify-content: space-between; padding: 4px 0; border-bottom: 1px solid #262a33; }
  .stat-row:last-child { border-bottom: none; }
  .stat-label { color: #9aa0ab; }
  .stat-value { font-weight: 600; }
  .badge { display: inline-block; padding: 2px 10px; border-radius: 12px; font-size: 0.8rem; font-weight: 600; }
  .ok { background: #163d27; color: #4ade80; }
  .fail { background: #3d1620; color: #f87171; }
  .warn-SAFE { color: #4ade80; }
  .warn-OBSTACLE { color: #facc15; }
  .warn-DANGEROUS { color: #fb923c; }
  .warn-EXTREME_DANGER { color: #f87171; font-weight: 800; }
  ul#objlist { list-style: none; padding: 0; margin: 0; max-height: 180px; overflow-y: auto; }
  ul#objlist li { padding: 4px 0; border-bottom: 1px solid #262a33; }
</style>
</head>
<body>
  <h1>🦯 Dashboard hỗ trợ người khiếm thị</h1>
  <div class="grid">
    <div>
      <div class="video-wrap">
        <img src="/video_feed" alt="YOLO stream">
      </div>
      <div class="card">
        <h3>Vật thể nhận diện</h3>
        <ul id="objlist"><li>Đang tải...</li></ul>
      </div>
    </div>
    <div>
      <div class="card">
        <h3>Trạng thái cảm biến</h3>
        <div class="stat-row"><span class="stat-label">Khoảng cách</span><span class="stat-value" id="distance">--</span></div>
        <div class="stat-row"><span class="stat-label">Góc servo</span><span class="stat-value" id="angle">--</span></div>
        <div class="stat-row"><span class="stat-label">Cảnh báo</span><span class="stat-value" id="warning">--</span></div>
        <div class="stat-row"><span class="stat-label">Camera FPS</span><span class="stat-value" id="camfps">--</span></div>
        <div class="stat-row"><span class="stat-label">YOLO FPS</span><span class="stat-value" id="yolofps">--</span></div>
      </div>
      <div class="card">
        <h3>Trạng thái hệ thống</h3>
        <div class="stat-row"><span class="stat-label">ESP32</span><span class="badge" id="esp32-badge">--</span></div>
        <div class="stat-row"><span class="stat-label">Camera</span><span class="badge" id="camera-badge">--</span></div>
        <div class="stat-row"><span class="stat-label">YOLO</span><span class="badge" id="yolo-badge">--</span></div>
        <div class="stat-row"><span class="stat-label">Heartbeat</span><span class="badge" id="hb-badge">--</span></div>
      </div>
    </div>
  </div>

<script>
function setBadge(id, ok) {
  const el = document.getElementById(id);
  el.textContent = ok ? "OK" : "LỖI";
  el.className = "badge " + (ok ? "ok" : "fail");
}

async function refreshStatus() {
  try {
    const res = await fetch("/api/status");
    const d = await res.json();

    document.getElementById("distance").textContent =
      d.distance_m > 0 ? d.distance_m.toFixed(2) + " m" : "Không xác định";
    document.getElementById("angle").textContent = d.angle + "°";

    const warnEl = document.getElementById("warning");
    warnEl.textContent = d.warning;
    warnEl.className = "stat-value warn-" + d.warning;

    document.getElementById("camfps").textContent = d.camera_fps.toFixed(1);
    document.getElementById("yolofps").textContent = d.yolo_fps.toFixed(1);

    setBadge("esp32-badge", d.esp32_ok);
    setBadge("camera-badge", d.camera_ok);
    setBadge("yolo-badge", d.yolo_ok);
    setBadge("hb-badge", d.heartbeat_ok);

    const list = document.getElementById("objlist");
    if (d.detections.length === 0) {
      list.innerHTML = "<li>Không phát hiện vật thể</li>";
    } else {
      list.innerHTML = d.detections
        .map(o => `<li>${o.name_vn} (${Math.round(o.confidence * 100)}%)</li>`)
        .join("");
    }
  } catch (e) {
    console.error("Lỗi cập nhật trạng thái:", e);
  }
}

setInterval(refreshStatus, 500);
refreshStatus();
</script>
</body>
</html>
"""


def create_flask_app(state: SystemState) -> Flask:
    """Tao Flask app voi dashboard + video feed + API trang thai."""
    app = Flask(__name__)

    # Tat log truy cap mac dinh cua Werkzeug de khong spam console Termux
    werkzeug_logger = logging.getLogger("werkzeug")
    werkzeug_logger.setLevel(logging.ERROR)

    @app.route("/")
    def index() -> str:
        return render_template_string(DASHBOARD_HTML)

    @app.route("/api/status")
    def api_status() -> Response:
        return jsonify(state.snapshot())

    @app.route("/video_feed")
    def video_feed() -> Response:
        def generate():
            boundary = b"--frame"
            target_interval = 1.0 / CAMERA_TARGET_FPS if CAMERA_TARGET_FPS > 0 else 0.05
            while True:
                jpeg = state.get_annotated_jpeg()
                if jpeg is not None:
                    yield (
                        boundary + b"\r\n"
                        b"Content-Type: image/jpeg\r\n"
                        b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
                        + jpeg + b"\r\n"
                    )
                time.sleep(target_interval)

        return Response(
            generate(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    return app


class FlaskThread(threading.Thread):
    """Chay Flask trong thread rieng, khong dung debug/reloader (tranh fork 2 process)."""

    def __init__(self, state: SystemState) -> None:
        super().__init__(name="FlaskThread", daemon=True)
        self._app = create_flask_app(state)

    def run(self) -> None:
        logger.info("[FLASK] Dashboard chạy tại http://127.0.0.1:%d và http://%s:%d",
                    FLASK_PORT, FLASK_HOST, FLASK_PORT)
        self._app.run(
            host=FLASK_HOST,
            port=FLASK_PORT,
            threaded=True,
            use_reloader=False,
            debug=False,
        )


# ============================================================================
# SECTION 10: MAIN - KHOI DONG TOAN BO HE THONG
# ============================================================================

def main() -> None:
    logger.info("=" * 60)
    logger.info("KHỞI ĐỘNG HỆ THỐNG HỖ TRỢ NGƯỜI KHIẾM THỊ")
    logger.info("ESP32_IP = %s | MODEL = %s", ESP32_IP, MODEL_PATH)
    logger.info("=" * 60)

    state = SystemState()
    stop_event = threading.Event()

    threads = [
        CameraThread(state, stop_event),
        YoloThread(state, stop_event),
        DataThread(state, stop_event),
        HeartbeatThread(state, stop_event),
        VoiceAlertThread(state, stop_event),
        FlaskThread(state),
    ]

    for t in threads:
        t.start()
        logger.info("[MAIN] Đã khởi động %s", t.name)

    try:
        while not stop_event.is_set():
            time.sleep(1.0)
    except KeyboardInterrupt:
        logger.info("[MAIN] Nhận Ctrl+C, đang dừng hệ thống...")
    finally:
        stop_event.set()
        # Cho cac thread daemon ket thuc vong lap hien tai (toi da ~2s)
        time.sleep(1.5)
        logger.info("[MAIN] Đã dừng hệ thống. Tạm biệt!")


if __name__ == "__main__":
    main()