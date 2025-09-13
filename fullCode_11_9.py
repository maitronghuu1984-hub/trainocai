# stream_detect_esp_flask.py
from flask import Flask, Response
from ultralytics import YOLO
import cv2, requests, time, os

# ========= CẤU HÌNH =========
MODEL_PATH = r"C:\Users\maitronghuu\PycharmProjects\OCBUUVANGAI\runs\detect\train3\weights\best.pt"

# ESP8266: cần có endpoint /send_gps (ví dụ GET)
ESP_HOST     = "http://192.168.0.109"        # ĐỔI theo IP in ở Serial của ESP
ESP_ENDPOINT = f"{ESP_HOST}/send_gps"
ESP_TIMEOUT  = 2.5

# Nhận dạng
TARGET_CLASSES = {"oc_buou_vang", "trung_oc"} # trùng tên class khi train
CONF_THRESHOLD = 0.70                         # chỉ vẽ & kích hoạt nếu conf >= 70%
REQUIRE_BOTH_CLASSES = False                  # False: chỉ cần 1 trong 2 lớp; True: phải thấy cả 2 lớp
TRIGGER_COOLDOWN_SEC = 15                     # chống spam lệnh ESP

# Camera & stream
CAMERA_INDEX = 0
FRAME_W, FRAME_H = 640, 480
TARGET_FPS = 20
JPEG_QUALITY = 80
SHOW_LOCAL = True                              # True: xem bằng cửa sổ OpenCV; False: chỉ MJPEG qua Flask

# ========= KHỞI TẠO =========
app = Flask(__name__)
model = YOLO(MODEL_PATH)
CLASS_NAMES = model.names  # dict: id -> name

last_trigger_time = 0.0

def send_to_esp():
    """Gửi lệnh sang ESP8266 (ví dụ để ESP tự đẩy GPS lên Firebase)."""
    try:
        r = requests.get(ESP_ENDPOINT, timeout=ESP_TIMEOUT)
        print(f"[ESP] Status: {r.status_code}")
    except Exception as e:
        print(f"[ESP] Lỗi gửi lệnh: {e}")

def should_trigger(highconf_labels: set) -> bool:
    """Quyết định kích hoạt dựa trên các nhãn mục tiêu đạt ngưỡng conf trong frame."""
    if REQUIRE_BOTH_CLASSES:
        return TARGET_CLASSES.issubset(highconf_labels)
    else:
        return len(TARGET_CLASSES.intersection(highconf_labels)) > 0

def process_and_annotate(frame):
    """
    Chạy YOLO và chỉ vẽ khung nếu:
      - Tên lớp thuộc TARGET_CLASSES
      - conf >= CONF_THRESHOLD
    Trả về (frame_annotated, trigger_bool)
    """
    results = model(frame, verbose=False)

    # Tập hợp những lớp MỤC TIÊU đã đạt ngưỡng trong frame
    highconf_labels = set()

    for res in results:
        boxes = res.boxes
        if boxes is None:
            continue

        for b in boxes:
            cls_id = int(b.cls[0])
            conf   = float(b.conf[0])
            name   = CLASS_NAMES.get(cls_id, str(cls_id))

            # CHỈ vẽ khung nếu đúng lớp mục tiêu & conf đủ ngưỡng
            if (name in TARGET_CLASSES) and (conf >= CONF_THRESHOLD):
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                label = f"{name} {conf:.2f}"
                color = (0, 255, 0)  # màu xanh cho mục tiêu đạt ngưỡng
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, max(y1-6, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

                highconf_labels.add(name)

            # Nếu không đạt điều kiện -> KHÔNG vẽ gì cả (bỏ qua)

    # Hiển thị info góc màn hình
    info = f"Thresh={int(CONF_THRESHOLD*100)}% | Cooldown={TRIGGER_COOLDOWN_SEC}s | RequireBoth={REQUIRE_BOTH_CLASSES}"
    cv2.putText(frame, info, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

    return frame, should_trigger(highconf_labels)

def gen_frames():
    """Trình phát MJPEG cho Flask."""
    global last_trigger_time

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_FPS,         TARGET_FPS)

    if not cap.isOpened():
        raise RuntimeError("Không mở được camera. Kiểm tra CAMERA_INDEX hoặc quyền truy cập camera.")

    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.02)
            continue

        frame, trigger_now = process_and_annotate(frame)

        # Chống spam lệnh
        now = time.time()
        if trigger_now and (now - last_trigger_time >= TRIGGER_COOLDOWN_SEC):
            print("[TRIGGER] Phát hiện mục tiêu >= 70%. Gửi lệnh ESP...")
            send_to_esp()
            last_trigger_time = now

        # Hiển thị local nếu cần
        if SHOW_LOCAL:
            cv2.imshow("OC_BUOU_VANG_AI", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC để thoát
                break

        # Mã hóa JPEG để stream
        ret, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        if not ret:
            continue
        jpg = buffer.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")

    cap.release()
    cv2.destroyAllWindows()

# ========= FLASK ROUTES =========
@app.route("/")
def index():
    # Trang đơn giản để xem stream
    html = """
    <html>
      <head><title>OC_BUOU_VANG_AI Stream</title></head>
      <body style="background:#111;color:#eee;font-family:Arial;">
        <h2>OC_BUOU_VANG_AI - MJPEG Stream</h2>
        <p>Chỉ vẽ box và kích hoạt ESP khi phát hiện <b>oc_buou_vang</b> hoặc <b>trung_oc</b> với conf ≥ 70%.</p>
        <img src="/video_feed" style="max-width:98%;border:2px solid #444;border-radius:8px;" />
      </body>
    </html>
    """
    return html

@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/health")
def health():
    return "ok"

# ========= MAIN =========
if __name__ == "__main__":
    # Chạy Flask; nếu chỉ xem cửa sổ OpenCV thì vẫn để SHOW_LOCAL=True và có thể không mở web.
    # Mở trong LAN: http://<IP_máy_chạy>:5000/
    app.run(host="0.0.0.0", port=5000, threaded=True)
