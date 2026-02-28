import os
import cv2
import numpy as np
import tensorflow as tf

DTYPE_STR = "float32"
DTYPE = np.float32

def define_data(imgs_folder, IMG_SIZE):
    imgs_data = []
    class_names = []

    for dir in os.listdir(imgs_folder):
        i = 0
        
        for file in os.listdir(os.path.join(imgs_folder,dir)):
            i+=1
            if i%500 == 0:
                print("loading {}th image for class {}".format(i,str(dir)))
            image_path = os.path.join(imgs_folder, dir, file)
            image = cv2.imread(image_path)
            if image is not None:
                image = cv2.cvtCOLOR(image,cv2.COLOR_BGR2RGB)
                image = cv2.resize(image,IMG_SIZE,interpolation = cv2.INTER_CUBIC)
                lab = cv2.cvtCOLOR(image,cv2.COLOR_RGB2LAB)
                L,A,B = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
                L = clahe.apply(L)
                lab = cv2.merge((L,A,B))
                image = cv2.cvtColor(lab,cv2.COLOR_LAB2RGB)
                image = image.astype(DTYPE_STR)
                imgs_data.append(image)
                class_names.append(dir)
    return np.array(imgs_data,DTYPE), class_names
    
def get_img_array(img_path,size):
    img = tf.keras.preprocessing.image.load_img(img_path,target_size=size)
    array = tf.keras.preprocessing.image.img_to_array(img)
    array = np.expand_dims(array,axis=0)
    return array