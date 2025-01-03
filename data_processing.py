
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator, img_to_array
from pathlib import Path
from PIL import Image
import numpy as nmp
import os

def positive_resize_normalize_and_augment(input_dir, output_dir, augment=True, augment_count=15):
    Path(output_dir).mkdir(parents=True, exist_ok=True) 

    data_augmentation = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    for img_path in Path(input_dir).glob('*'):
        with Image.open(img_path) as img:
            img = img.resize((128, 128))
            img_array = nmp.array(img) / 255.0
            img = Image.fromarray((img_array * 255).astype(nmp.uint8))      
            img = img.convert('RGB')
            output_path = Path(output_dir) / img_path.name
            img.save(output_path)
            print(f"{img} başarıyla kaydedildi")

            if augment == True:
                x = img_to_array(img)
                x = x.reshape((1,) + x.shape) 

                i = 1
                for _ in data_augmentation.flow(x, batch_size=1, save_to_dir=output_dir, save_format='png'):
                    if i <= augment_count:
                        i += 1
                    else:
                        break

abs_path = os.getcwd()

positive_input_dir = os.path.join(abs_path, "Dataset", "Positive")
positive_output_dir = os.path.join(abs_path, "Dataset", "Re-Positive")
positive_resize_normalize_and_augment(positive_input_dir, positive_output_dir)


def negative_resize_normalize_and_move(input_dir, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)     

    negative_lst = os.listdir(input_dir)
    negative_num = len(negative_lst)  
    
    for i in range(negative_num):                                  
        dir_path = Path(input_dir) / negative_lst[i]                   
        for j, img_path in enumerate(dir_path.glob('*')):           
            if j >= 50:
                break
            with Image.open(img_path) as img:
                img = img.resize((128, 128))
                img_array = nmp.array(img) / 255.0
                img = Image.fromarray((img_array * 255).astype(nmp.uint8))
                img = img.convert('RGB')
                output_file = Path(output_dir) / f"negative_{i}_{j}.png"
                img.save(output_file)
                print(f"saved succefuly: {output_file}")

negative_input_dir = os.path.join(abs_path, "Dataset", "Negative")
negative_output_dir = os.path.join(abs_path, "Dataset", "Re-Negative")
negative_resize_normalize_and_move(negative_input_dir, negative_output_dir)

print(f"len of Re-Positive: ", len(list(Path( os.path.join(abs_path, "Dataset", "Re-Positive")).glob('*'))))
print(f"len of Re-Negative: ", len(list(Path( os.path.join(abs_path, "Dataset", "Re-Negative")).glob('*'))))

