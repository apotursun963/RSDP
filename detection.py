
from keras._tf_keras.keras.models import load_model
import numpy as nmp
import cv2, os

abs_path = os.getcwd()
model = load_model(os.path.join(abs_path, "CNN-MODEL.h5"))

def process_image(image):
    img = cv2.resize(image, (32,32))
    img = img / 255.0
    img = nmp.expand_dims(img, axis=0)
    return (img)

threshold = 0.5
vampirt_vid = os.path.join(abs_path, "vid", "RS.mp4")
not_vampir_vid = os.path.join(abs_path, "vid", "!RS.mp4")
open_cmr = cv2.VideoCapture(not_vampir_vid)

while True:
    ret, frame = open_cmr.read()       
    process_frame = process_image(frame)
    tahmin = model.predict(process_frame)

    label = nmp.argmax(tahmin, axis=1)
    confidence = nmp.max(tahmin)
    predicted_class = tahmin[label[0]]

    cv2.putText(frame, f'Label: {predicted_class} Confidence: {confidence:.2f}', 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    if predicted_class > threshold:
        cv2.putText(frame, 'ilaci sik',
                    (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

open_cmr.release()
cv2.destroyAllWindows()
