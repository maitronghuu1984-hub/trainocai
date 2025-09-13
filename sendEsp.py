from ultralytics import YOLO
import cv2
import requests
import time

# ====== Cấu hình ======
MODEL_PATH = r"C:\Users\maitronghuu\PycharmProjects\OCBUUVANGAI\runs\detect\train3\weights\best.pt"
ESP_HOST   = "http://192.168.0.109"        # <-- ĐỔI IP NÀY theo IP in ra ở Serial của ESP
ESP_ENDPOINT = f"{ESP_HOST}/send_gps"
TARGET_CLASSES = {"oc_buou_vang", "trung_oc"}
CONF_THRESHOLD = 0.5
TRIGGER_COOLDOWN_SEC = 15                   # chống gọi dồn dập
# ======================

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Không thể mở camera")
    raise SystemExit

print("🎥 Đang nhận diện từ webcam... Nhấn Q để thoát")

last_trigger_time = 0

def trigger_gps_send():
    try:
        r = requests.get(ESP_ENDPOINT, timeout=5)
        if r.status_code == 200:
            print("✅ Đã gửi lệnh cho ESP8266: yêu cầu gửi GPS.")
        else:
            print("⚠️ ESP8266 phản hồi mã:", r.status_code, r.text)
    except Exception as e:
        print("❌ Lỗi gửi lệnh đến ESP8266:", e)

while True:
    ok, frame = cap.read()
    if not ok:
        break

    # Chạy YOLO (ẩn log cho gọn)
    results = model(frame, verbose=False)

    found_snail = False

    for r in results:
        boxes = r.boxes
        if boxes is not None and len(boxes) > 0:
            # Duyệt từng box
            for cls_id, conf in zip(boxes.cls.tolist(), boxes.conf.tolist()):
                cls_id = int(cls_id)
                conf = float(conf)
                class_name = model.names.get(cls_id, str(cls_id))
                if class_name in TARGET_CLASSES and conf >= CONF_THRESHOLD:
                    found_snail = True
                    break

        # Vẽ khung & nhãn để xem trực tiếp
        im_array = r.plot()
        cv2.imshow("Nhận dạng trực tiếp", im_array)

    # Nếu phát hiện và qua cooldown -> gọi ESP
    now = time.time()
    if found_snail and (now - last_trigger_time >= TRIGGER_COOLDOWN_SEC):
        print("🟡 Phát hiện ốc/trứng ốc đạt ngưỡng, gửi lệnh ESP lấy GPS...")
        trigger_gps_send()
        last_trigger_time = now

    if cv2.waitKey(3) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
