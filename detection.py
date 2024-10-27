
from keras._tf_keras.keras.models import load_model
from PIL import Image
import numpy as nmp
import cv2

model = load_model("C:\\Users\\90507\\OneDrive\\Masaüstü\\Coding\\Deep Learning\\Object Recognition Projects\\RSDP\\CNN-MODEL.h5")

def process_image(image):
    img = cv2.resize(image, (32,32))
    img = img / 255.0
    img = nmp.expand_dims(img, axis=0)
    return (img)

threshold = 0.5
vampirt_vid = "C:\\Users\\90507\\OneDrive\\Masaüstü\\Coding\\Deep Learning\\Object Recognition Projects\\RSDP\\vampir_vid.mp4"
not_vampir_vid = "C:\\Users\\90507\\OneDrive\\Masaüstü\\Coding\\Deep Learning\\Object Recognition Projects\\RSDP\\vampir_not_vid.mp4"
kamera_ac = cv2.VideoCapture(not_vampir_vid)

while True:
    ret, frame = kamera_ac.read()       

    process_frame = process_image(frame)

    tahmin = model.predict(process_frame)

    label = nmp.argmax(tahmin, axis=1)
    confidence = nmp.max(tahmin)
    predicted_class = tahmin[label[0]]

    # Sonuçları ekranda gösteriyor
    cv2.putText(frame, f'Label: {predicted_class} Confidence: {confidence:.2f}', 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    if predicted_class > threshold:
        cv2.putText(frame, 'ilaci sik',        # buraya yazılacak kod robotun ilacı sıkma işlevi olmalı
                    (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):           # 'q' tuşuna basıldığında frame'de çıkılacak
        break

kamera_ac.release()
cv2.destroyAllWindows()


# test
def pre_process_img(path):
    with Image.open(path) as img:
        img = img.convert('RGB')
        img = img.resize((32,32))
        img_array = nmp.array(img) / 255.0
        img_array = nmp.expand_dims(img_array, axis=0)
    return (img_array)

def predict_image(image_path):
    img_array = pre_process_img(image_path)
    prediction = model.predict(img_array)
    return (prediction)


images_path = (
    "C:\\Users\\90507\\OneDrive\\Masaüstü\\Coding\\Deep Learning\\Object Recognition Projects\\RSDP\\test verileri\\vampir.png", 
    "C:\\Users\\90507\\OneDrive\\Masaüstü\\Coding\\Deep Learning\\Object Recognition Projects\\RSDP\\test verileri\\vampir2.png", 
    "C:\\Users\\90507\\OneDrive\\Masaüstü\\Coding\\Deep Learning\\Object Recognition Projects\\RSDP\\test verileri\\vampir3.png",
    "C:\\Users\\90507\\OneDrive\\Masaüstü\\Coding\\Deep Learning\\Object Recognition Projects\\RSDP\\test verileri\\vampir4.png",
    "C:\\Users\\90507\\OneDrive\\Masaüstü\\Coding\\Deep Learning\\Object Recognition Projects\\RSDP\\test verileri\\vampir5.png",
    "C:\\Users\\90507\\OneDrive\\Masaüstü\\Coding\\Deep Learning\\Object Recognition Projects\\RSDP\\test verileri\\vampir_değil.png",
    "C:\\Users\\90507\\OneDrive\\Masaüstü\\Coding\\Deep Learning\\Object Recognition Projects\\RSDP\\test verileri\\vampir_değil2.png",
    "C:\\Users\\90507\\OneDrive\\Masaüstü\\Coding\\Deep Learning\\Object Recognition Projects\\RSDP\\test verileri\\vampir_değil3.png",
    "C:\\Users\\90507\\OneDrive\\Masaüstü\\Coding\\Deep Learning\\Object Recognition Projects\\RSDP\\test verileri\\vampir_değil4.png",
    "C:\\Users\\90507\\OneDrive\\Masaüstü\\Coding\\Deep Learning\\Object Recognition Projects\\RSDP\\test verileri\\vampir_değil5.png",
)

for img_path in images_path:
    prediction = predict_image(img_path)
    print(f"Vampir kelebek olma ihtimali: {prediction[0][0]}")
