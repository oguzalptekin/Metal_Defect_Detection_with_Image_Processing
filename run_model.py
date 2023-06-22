from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
import random
from matplotlib.patches import Rectangle
from lxml import etree
import os
import tensorflow as tf

def dir_prepare():
    os.listdir('./images/images')
    os.listdir('./label/label')

    image_path = glob.glob('./images/images/*/*.jpg')
    len(image_path)

    xmls_path = glob.glob('./label/label/*.xml')
    len(xmls_path)

    #xml_name extraction
    xmls_train = [p.split('\\')[-1].split('.')[0] for p in xmls_path]

    #img_name extraction
    imgs_train = [img for img in image_path if (img.split('\\')[-1].split)('.jpg')[0] in xmls_train]

    #check the image to label sorts
    xmls_path.sort(key=lambda x:x.split('\\')[-1].split('.xml')[0])
    imgs_train.sort(key=lambda x:x.split('\\')[-1].split('.jpg')[0])

    #labels names
    names = [x.split("\\")[-2] for x in imgs_train]

    names = pd.DataFrame(names,columns=['Types'])

    #onehot for mutiple classes
    from sklearn.preprocessing import LabelBinarizer

    Class = names['Types'].unique()
    Class_dict = dict(zip(Class, range(1,len(Class)+1)))
    names['str'] = names['Types'].apply(lambda x: Class_dict[x])
    lb = LabelBinarizer()
    lb.fit(list(Class_dict.values()))
    transformed_labels = lb.transform(names['str'])
    y_bin_labels = []

    for i in range(transformed_labels.shape[1]):
        y_bin_labels.append('str' + str(i))
        names['str' + str(i)] = transformed_labels[:, i]

    return Class_dict




def preprocess_image(img_path):
    # Read and decode the image
    image = tf.io.read_file(img_path)
    image = tf.image.decode_jpeg(image, channels=3)

    # Convert RGB image to grayscale
    image_gray = tf.image.rgb_to_grayscale(image)

    # Apply Gaussian blur for noise reduction
    image_blur = tf.image.convert_image_dtype(image_gray, dtype=tf.float32)
    image_blur = tf.image.resize(image_blur, [224, 224])
    image_blur = tf.image.random_flip_left_right(image_blur)  # Optional data augmentation
    image_blur = tf.image.random_flip_up_down(image_blur)  # Optional data augmentation
    image_blur = tf.image.random_brightness(image_blur, max_delta=0.2)  # Optional data augmentation
    image_blur = tf.clip_by_value(image_blur, 0.0, 1.0)  # Clip values to [0, 1]

    # Convert grayscale image back to RGB
    image_rgb = tf.image.grayscale_to_rgb(image_blur)

    # Normalize pixel values to [-1, 1]
    image_normalized = image_rgb * 2.0 - 1.0

    return image_normalized


def draw_plot(images, predictions):
    Class_dict = dir_prepare()
    species_dict = {v: k for k, v in Class_dict.items()}

    plt.figure(figsize=(10, 24))
    for i in range(len(images)):
        img = images[i]
        preds = predictions[i]

        plt.subplot(len(images), 1, i + 1)
        plt.imshow(keras.preprocessing.image.array_to_img(img))

        pred_imglist = []
        pred_imglist.append(species_dict[np.argmax(preds[-1]) + 1])
        plt.title(pred_imglist)

        xmin, ymin, xmax, ymax = preds[0] * 224, preds[1] * 224, preds[2] * 224, preds[3] * 224
        rect = Rectangle((xmin, ymin), (xmax - xmin), (ymax - ymin), fill=False, color='r')
        ax = plt.gca()
        ax.axes.add_patch(rect)
    return plt



dir_prepare()
img_path = "./static/brokenmetal.jpg"
processed_image = preprocess_image(img_path)
model = keras.models.load_model("defect_model.h5")
predictions = model.predict(tf.expand_dims(processed_image, axis=0))
draw_plot([processed_image], [predictions])

#done