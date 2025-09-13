from ultralytics import YOLO
from PIL import Image

# Create a new YOLO model from scratch


# Load a pretrained YOLO model (recommended for training)
model = YOLO("C:\\Users\\AD\\PycharmProjects\OCBUUVANGAI\\runs\detect\\train3\\weights\\best.pt")

# Perform object detection on an image using the model
results = model("C:\\Users\\AD\\PycharmProjects\\OCBUUVANGAI\\trungocvang1.jpg")

for r in results:
    print(r.boxes)
    im_array=r.plot()
    im=Image.fromarray(im_array[..., ::-1])
    im.show()
    im.save("ketqua.png")