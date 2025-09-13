from ultralytics import YOLO
from PIL import Image
import cv2

# Load mô hình đã huấn luyện (YOLOv8)
model = YOLO(r"C:\\Users\\maitronghuu\\PycharmProjects\\OCBUUVANGAI\\runs\\detect\\train3\\weights\\best.pt")

# Mở webcam (0 = camera mặc định, nếu dùng camera ngoài thì dùng 1, 2...)
cap = cv2.VideoCapture(0)

# Kiểm tra nếu không mở được camera
if not cap.isOpened():
    print("❌ Không thể mở camera")
    exit()

print("🎥 Đang nhận diện từ webcam... Nhấn Q để thoát")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Nhận diện bằng YOLO
    results = model(frame)

    for r in results:
        # Vẽ khung nhận diện
        im_array = r.plot()

        # Hiển thị bằng OpenCV
        cv2.imshow("Nhận dạng trực tiếp", im_array)

    # Nhấn Q để thoát
    if cv2.waitKey(3) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
