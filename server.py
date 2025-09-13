from flask import Flask, Response
import cv2
from ultralytics import YOLO

# Khởi tạo Flask app
app = Flask(__name__)

# Load mô hình YOLOv8 đã huấn luyện
model = YOLO(r"C:\Users\maitronghuu\PycharmProjects\OCBUUVANGAI\runs\detect\train3\weights\best.pt")

# Mở webcam
cap = cv2.VideoCapture(0)  # Hoặc 1 nếu dùng camera ngoài

def gen_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Dự đoán bằng YOLO
        results = model(frame)

        # Vẽ kết quả lên ảnh
        for r in results:
            frame = r.plot()  # Gán khung nhận diện vào frame

        # Mã hóa JPEG để stream
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Gửi từng frame
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Route phát video
@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Giao diện đơn giản
@app.route('/')
def index():
    return '''
    <h2>📸 Nhận diện ốc bươu vàng bằng YOLOv8</h2>
    <img src="/video" width="100%">
    '''

if __name__ == '__main__':
    # Chạy server trên mọi IP, cổng 5000
    app.run(host='0.0.0.0', port=5000)