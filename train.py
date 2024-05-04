import tensorflow as tf
from tensorflow.keras import layers, models, datasets
import numpy as np


# optimized lenet-5 98% accurate
def build_lenet5(input_shape=(32, 32, 1)):
    model = models.Sequential([
        layers.Conv2D(filters=6, kernel_size=(5, 5), activation='relu', input_shape=input_shape, padding='valid', kernel_initializer='he_uniform'),

        layers.AveragePooling2D(pool_size=(2, 2), strides=2),
        layers.BatchNormalization(),

        layers.Conv2D(filters=16, kernel_size=(5, 5), activation='relu', padding='valid', kernel_initializer='he_uniform'),

        layers.AveragePooling2D(pool_size=(2, 2), strides=2),
        layers.BatchNormalization(),

        layers.Conv2D(filters=120, kernel_size=(5, 5), activation='relu', kernel_initializer='he_uniform'),
        layers.BatchNormalization(),

        layers.Flatten(),

        layers.Dense(units=84, activation='relu', kernel_initializer='he_uniform'),
        layers.BatchNormalization(),

        layers.Dense(units=10, activation='softmax')
    ])
    

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# load data
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

# process data 
train_images = np.pad(train_images, ((0,0),(2,2),(2,2)), mode='constant', constant_values=0)
test_images = np.pad(test_images, ((0,0),(2,2),(2,2)), mode='constant', constant_values=0)
train_images = train_images[..., np.newaxis] / 255.0 
test_images = test_images[..., np.newaxis] / 255.0    

train_labels = tf.keras.utils.to_categorical(train_labels, 10)
test_labels = tf.keras.utils.to_categorical(test_labels, 10)


# build optimized model
model = build_lenet5()

model.summary()

# train
model.fit(train_images, train_labels, epochs=10, validation_split=0.1)

# result
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test Accuracy: {test_acc*100:.2f}%")