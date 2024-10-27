Ricania Simulans Detection using CNN
-------------------------------------
This project use of Convolutional Neural Networks (CNNs) to detect vampire butterflies from images. It involves data preprocessing, model training, and object detection, utilizing a dataset with positive (vampire butterflies) and negative (other butterfly species) samples. The dataset contains subfolders for both positive and negative images, and the preprocessing scripts resize, normalize, and augment these images. The model creation script defines and trains a CNN model using the preprocessed images. Finally, the trained model classifies new images and video frames, offering real-time object detection and video processing functionality.

# Model
```python 
model = Sequential([

    Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=(32,32,3)),
    MaxPooling2D(pool_size=(2,2)),
    Dropout(0.1),

    Conv2D(filters=64, kernel_size=(3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Dropout(0.1),

    Conv2D(filters=32, kernel_size=(3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Dropout(0.1),

    Flatten(),
    Dense(units=256, activation='relu'),
    Dropout(0.3),

    Dense(units=128, activation='relu'),
    Dropout(0.3),

    Dense(units=64, activation='relu'),
    Dropout(0.3),
    Dense(units=1, activation='sigmoid')
])
```

R1esults
-------
![Figure_1](https://github.com/user-attachments/assets/e5e32325-6485-4bfe-8fbb-fba5c4465355)
