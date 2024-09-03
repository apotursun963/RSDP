
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from sklearn.model_selection import train_test_split
import numpy as nmp
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

def labelling_data(positive_dir, negative_dir):
    images, labels = [], []

    def process_img(directroy_path, label):
        for img_path in Path(directroy_path).glob('*'):
            with Image.open(img_path) as img:
                img = img.resize((32, 32))
                img_array = nmp.array(img) / 255.0
                images.append(img_array)        
                labels.append(label)

    process_img(positive_dir, 1) 
    process_img(negative_dir, 0) 

    x = nmp.array(images)
    y = nmp.array(labels)
    indices = nmp.arange(x.shape[0])
    nmp.random.shuffle(indices)
    x = x[indices]
    y = y[indices]

    return (x,y)


positive_folder = "C:\\Users\\90507\\OneDrive\\Masaüstü\\Coding\\DL\\Nesne Tanıma Projeleri\\VKTP\\Dataset\\Re-Positive"
negative_folder = "C:\\Users\\90507\\OneDrive\\Masaüstü\\Coding\\DL\\Nesne Tanıma Projeleri\\VKTP\\Dataset\\Re-Negative"

x, y = labelling_data(positive_folder, negative_folder)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.01)

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=(32,32,3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.1))
model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.1))
model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.1))
model.add(Flatten())
model.add(Dense(units=256, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(units=128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(units=64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(optimizer='adam',
              loss='binary_crossentropy', 
              metrics=['accuracy'])

train = model.fit(x_train, y_train,
          batch_size=15,
          epochs=10,
          validation_split=0.1)

test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test doğruluğu: {test_acc} | Test kaybı: {test_loss}")

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train.history['loss'], label='Training Loss')
plt.plot(train.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss')

plt.subplot(1, 2, 2)
plt.plot(train.history['accuracy'], label='Training Accuracy')
plt.plot(train.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy')

plt.show()

# modeli kaydet
model.save("CNN-MODEL.h5")
