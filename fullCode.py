# stream_detect_esp_flask.py
from flask import Flask, Response
from ultralytics import YOLO
import cv2, requests, time, threading, os

# ========= CẤU HÌNH =========
MODEL_PATH = r"C:\Users\maitronghuu\PycharmProjects\OCBUUVANGAI\runs\detect\train3\weights\best.pt"

# ESP8266: cần có endpoint /send_gps
ESP_HOST   = "http://192.168.0.109"          # ĐỔI theo IP in ở Serial của ESP
ESP_ENDPOINT = f"{ESP_HOST}/send_gps"        # ESP xử lý và đẩy GPS lên Firebase

# Nhận dạng
TARGET_CLASSES = {"oc_buou_vang", "trung_oc"} # trùng tên class khi train
CONF_THRESHOLD = 0.5
TRIGGER_COOLDOWN_SEC = 15                     # chống spam lệnh ESP

# Camera & stream
CAMERA_INDEX = 0
FRAME_W, FRAME_H = 640, 480
TARGET_FPS = 20                               # 15–20 cho mượt
JPEG_QUALITY = 80
SHOW_LOCAL_WINDOW = False                     # True nếu muốn xem cửa sổ OpenCV

# Flask
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000
# ============================

# ======== KHỞI TẠO ========
app = Flask(__name__)
model = YOLO(MODEL_PATH)

last_frame_jpg = None
last_trigger_time = 0.0
frame_lock = threading.Lock()
stop_event = threading.Event()

def trigger_gps_send():
    """Gọi ESP8266 yêu cầu gửi GPS lên Firebase."""
    try:
        r = requests.get(ESP_ENDPOINT, timeout=5)
        if r.status_code == 200:
            print("✅ Lệnh gửi GPS đã đến ESP8266.")
        else:
            print("⚠️ ESP8266 phản hồi:", r.status_code, r.text)
    except Exception as e:
        print("❌ Lỗi gọi ESP8266:", e)

def detection_loop():
    """Đọc camera → YOLO → cập nhật khung hình stream và kích hoạt ESP theo điều kiện."""
    global last_frame_jpg, last_trigger_time

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("❌ Không thể mở camera")
        stop_event.set()
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

    print("🎥 Đang nhận diện... (stream tại /video). Nhấn Ctrl+C để dừng.")

    try:
        while not stop_event.is_set():
            t0 = time.time()
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.05)
                continue

            results = model(frame, verbose=False)
            found_snail = False
            im_vis = frame

            for r in results:
                boxes = r.boxes
                if boxes is not None and len(boxes) > 0:
                    for cls_id, conf in zip(boxes.cls.tolist(), boxes.conf.tolist()):
                        cls_id = int(cls_id); conf = float(conf)
                        class_name = model.names.get(cls_id, str(cls_id))
                        if class_name in TARGET_CLASSES and conf >= CONF_THRESHOLD:
                            found_snail = True
                            break
                im_vis = r.plot()  # frame có box/label

            # Cooldown để không gọi ESP dồn dập
            now = time.time()
            if found_snail and (now - last_trigger_time >= TRIGGER_COOLDOWN_SEC):
                print("🟡 Phát hiện ốc/trứng ốc → gọi ESP gửi GPS lên Firebase...")
                trigger_gps_send()
                last_trigger_time = now

            # Mã hoá JPEG để stream
            ok, buf = cv2.imencode(".jpg", im_vis, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            if ok:
                with frame_lock:
                    last_frame_jpg = buf.tobytes()

            if SHOW_LOCAL_WINDOW:
                cv2.imshow("Nhận dạng trực tiếp", im_vis)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    stop_event.set()
                    break

            # Giới hạn FPS
            elapsed = time.time() - t0
            sleep = max(0.0, (1.0 / TARGET_FPS) - elapsed)
            if sleep > 0:
                time.sleep(sleep)
    finally:
        cap.release()
        if SHOW_LOCAL_WINDOW:
            cv2.destroyAllWindows()
        print("🛑 Đã đóng camera.")

@app.route("/")
def index():
    return '''<h3>📸 YOLO + ESP8266 GPS → Firebase</h3><img src="/video" width="100%">'''

@app.route("/video")
def video():
    def gen():
        boundary = b"--frame"
        while not stop_event.is_set():
            with frame_lock:
                frame = last_frame_jpg
            if frame is None:
                time.sleep(0.02)
                continue
            yield (
                boundary + b"\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" +
                frame + b"\r\n"
            )
            time.sleep(1.0 / TARGET_FPS)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/health")
def health():
    return {"ok": True, "model": os.path.basename(MODEL_PATH)}, 200

if __name__ == "__main__":
    # Chạy nhận dạng ở thread riêng
    t = threading.Thread(target=detection_loop, daemon=True)
    t.start()

    try:
        # Mở server MJPEG cho app
        app.run(host=FLASK_HOST, port=FLASK_PORT, threaded=True)
    finally:
        stop_event.set()
        t.join(timeout=2.0)
        print("✅ Đã dừng server.")
