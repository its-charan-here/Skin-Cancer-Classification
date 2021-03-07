from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import roc_curve
from tensorflow.keras import layers
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_hub as hub
import math
import tensorflow as tf
import os
import shutil
import random


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


train_examples = 20225
test_examples = 2551
validation_examples = 2555
img_height = img_width = 224
batch_size = 32
epochs = 10


'''
#Nasnet

model = keras.Sequential([
    hub.KerasLayer("https://tfhub.dev/google/imagenet/nasnet_mobile/feature_vector/4",
                  trainable=True),
    layers.Dense(1,activation="sigmoid") # sigmoid because we have only 2 classes
])
model.build([None,img_height,img_width,3])
print(model.summary())
'''

# EfficientNet

model_url = "https://tfhub.dev/tensorflow/efficientnet/b0/classification/1"
model = keras.Sequential([
    hub.KerasLayer(model_url,trainable=True),
    layers.Dense(1,activation="sigmoid") # sigmoid because we have only 2 classes
])

model.build([None, img_height, img_width, 3])
print(model.summary())


# model = keras.models.load_model("isic_model/")  # use this line of code if you want to resume training
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=15,
    zoom_range=(0.95, 0.95),
    horizontal_flip=True,
    vertical_flip=True,
    data_format="channels_last",
    dtype=tf.float32,
)

validation_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    dtype=tf.float32,
)
test_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    dtype=tf.float32,
)

train_gen = train_datagen.flow_from_directory(
    "data/train/",
    target_size=(img_height, img_width),
    batch_size=batch_size,
    color_mode="rgb",
    class_mode="binary",
    shuffle=True,
    seed=123,
)

validation_gen = validation_datagen.flow_from_directory(
    "data/validation/",
    target_size=(img_height, img_width),
    batch_size=batch_size,
    color_mode="rgb",
    class_mode="binary",
    shuffle=True,
    seed=123,
)

test_gen = test_datagen.flow_from_directory(
    "data/test/",
    target_size=(img_height, img_width),
    batch_size=batch_size,
    color_mode="rgb",
    class_mode="binary",
    shuffle=True,
    seed=123,
)


# since our dataset is skewed we have to introduce a few metrics which are inbuilt in Keras to handle data

METRICS = [
    keras.metrics.BinaryAccuracy(name="accuracy"),
    keras.metrics.Precision(name="precision"),
    keras.metrics.Recall(name="recall"),
    keras.metrics.AUC(name="auc"),
]


# compile model
model.compile(
    optimizer=keras.optimizers.Adam(lr=3e-4),
    loss=[keras.losses.BinaryCrossentropy(from_logits=False)],
    metrics=METRICS,
)

model.fit(
    train_gen,
    epochs=epochs,
    verbose=1,
    steps_per_epoch=train_examples // batch_size,
    validation_data=validation_gen,
    validation_steps=validation_examples // batch_size,
    callbacks=[keras.callbacks.ModelCheckpoint("isic_model")],
)


# plotting a ROC curve (it predicts the true positive rate vs false positive rate)
def plot_roc(labels, data):
    predictions = model.predict(data)
    fp, tp, _ = roc_curve(labels, predictions)
    plt.plot(100 * fp, 100 * tp)
    plt.xlabel("False positives [%]")
    plt.ylabel("True positives [%]")
    plt.show()


# the following lines of code are just to get all the labels from the test set to a single np array
test_labels = np.array([])
num_batches = 0

for _, y in test_gen:
    test_labels = np.append(test_labels, y)
    num_batches += 1
    if num_batches == math.ceil(test_examples / batch_size):
        break

plot_roc(test_labels, test_gen)


# evaluate the model

model.evaluate(validation_gen, verbose=1)
model.evaluate(test_gen, verbose=1)
