from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from collections import Counter
import tensorflow.keras as keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, InputLayer
from keras.layers import Activation, MaxPooling2D, Dropout, Flatten, Reshape
from keras.utils import to_categorical
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def categorical_to_numpy(labels_in):
    labels = []
    for label in labels_in:
        if label == 'dog':
            labels.append(np.array([1, 0]))
        else:
            labels.append(np.array([0, 1]))
    return np.array(labels)

def one_hot_encoding(input):
    output = np.zeros((input.size, input.max()+1))
    output[np.arange(input.size), input] = 1
    return output

def load_data():
    !wget -O cifar_data https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%201%20-%205/Session%204%20_%205%20-%20Neural%20Networks%20_%20CNN/dogs_v_roads
    import pickle
    data_dict = pickle.load(open("cifar_data", "rb"))
    data = data_dict['data']
    labels = data_dict['labels']
    return data, labels

def plot_one_image(data, labels, img_idx):
    from google.colab.patches import cv2_imshow
    import cv2
    import matplotlib.pyplot as plt
    my_img = data[img_idx, :].reshape([32, 32, 3]).copy()
    my_label = labels[img_idx]
    print(f'label: {my_label}')
    fig, ax = plt.subplots(1, 1)
    img = ax.imshow(my_img.astype('uint8'), extent=[-1, 1, -1, 1])
    x_label_list = [0, 8, 16, 24, 32]
    y_label_list = [0, 8, 16, 24, 32]
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.set_xticklabels(x_label_list)
    ax.set_yticklabels(y_label_list)
    fig.show(img)

def logits_to_one_hot_encoding(input):
    output = np.zeros_like(input, dtype=int)
    output[np.arange(len(input)), np.argmax(input, axis=1)] = 1
    return output

class CNNClassifier:
    def __init__(self, num_epochs=30, layers=4, dropout=0.5):
        self.num_epochs = num_epochs
        self.layers = layers
        self.dropout = dropout
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Reshape((32, 32, 3)))
        for i in range(self.layers):
            model.add(Conv2D(32, (3, 3), padding='same'))
            model.add(Activation('relu'))
        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(self.dropout))
        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(self.dropout))
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dropout(self.dropout))
        model.add(Dense(2))
        model.add(Activation('softmax'))
        opt = keras.optimizers.legacy.RMSprop(learning_rate=0.0001, decay=1e-6)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        return model

    def fit(self, *args, **kwargs):
        return self.model.fit(*args, epochs=self.num_epochs, batch_size=10, verbose=2, **kwargs)

    def predict(self, *args, **kwargs):
        predictions = self.model.predict(*args, **kwargs)
        return logits_to_one_hot_encoding(predictions)

    def predict_proba(self, *args, **kwargs):
        predictions = self.model.predict(*args, **kwargs)
        return predictions

    def score(self, X, y):
        predictions = self.predict(X)
        return accuracy_score(y, predictions)

    def __getattr__(self, name):
        if name != 'predict' and name != 'predict_proba':
            return getattr(self.model, name)
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

def plot_acc(history, ax=None, xlabel='Epoch #'):
    history = history.history
    history.update({'epoch': list(range(len(history['val_accuracy'])))})
    history = pd.DataFrame.from_dict(history)
    best_epoch = history.sort_values(by='val_accuracy', ascending=False).iloc[0]['epoch']
    if not ax:
        f, ax = plt.subplots(1, 1)
    sns.lineplot(x='epoch', y='val_accuracy', data=history, label='Validation', ax=ax)
    sns.lineplot(x='epoch', y='accuracy', data=history, label='Training', ax=ax)
    ax.axhline(0.5, linestyle='--', color='red', label='Chance')
    ax.axvline(x=best_epoch, linestyle='--', color='green', label='Best Epoch')
    ax.legend(loc=7)
    ax.set_ylim([0.4, 1])
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Accuracy (Fraction)')
    plt.show()

def model_to_string(model):
    import re
    stringlist = []
    model.summary(print_fn=lambda x: stringlist.append(x))
    sms = "\n".join(stringlist)
    sms = re.sub('_\d\d\d', '', sms)
    sms = re.sub('_\d\d', '', sms)
    sms = re.sub('_\d', '', sms)
    return sms

data_raw, labels_raw = load_data()
data = data_raw.astype(float)
labels = categorical_to_numpy(labels_raw)
X_train, X_test, y_train, y_test = model_selection.train_test_split(data, labels, test_size=0.2, random_state=1)

print(data.shape)

plot_one_image(data_raw, labels_raw, 300)

cnn = CNNClassifier(num_epochs=5, layers=5, dropout=0.5)
cnn.fit(X_train, y_train)

print(cnn.score(X_test, y_test))

cnn = CNNClassifier(num_epochs=5, layers=5, dropout=0.5)
history = cnn.fit(X_train, y_train, validation_data=(X_test, y_test))
plot_acc(history)

print("CNN Testing Set Score:")
print(cnn.score(X_test, y_test))

model_1 = Sequential()
model_1.add(InputLayer(input_shape=(3,)))
model_1.add(Dense(4, activation='relu'))
model_1.add(Dense(2, activation='softmax'))
model_1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model_1.predict([[14, 18, 5]])

model_2 = Sequential()
model_2.add(InputLayer(input_shape=(3,)))
model_2.add(Dense(4, activation='relu'))
model_2.add(Dense(4, activation='relu'))
model_2.add(Dense(1, activation='relu'))
model_2.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
model_2.predict([[12, 6, 6]])

model_3 = Sequential()
model_3.add(InputLayer(input_shape=(3072,)))
model_3.add(Dense(32, activation='relu'))
model_3.add(Dense(16, activation='relu'))
model_3.add(Dense(2, activation='softmax'))
model_3.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model_3.predict(X_train)

model = Sequential()
model.add(Reshape((32, 32, 3)))
model.add(Conv2D(32, (3, 3), padding='same'))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), padding='same'))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), padding='same'))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Dense(2))
model.add(Activation('softmax'))

opt = keras.optimizers.legacy.RMSprop(learning_rate=0.0001, decay=1e-6)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=70)
plot_acc(history)

model.summary()

import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from google.colab.patches import cv2_imshow
import cv2
import matplotlib.pyplot as plt

try:
    road_model = model
    road_saved = True
except NameError:
    road_saved = False

IMG_SHAPE = 150
_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
zip_dir = tf.keras.utils.get_file('cats_and_dogs_filterted.zip', origin=_URL, extract=True)
base_dir = os.path.join(os.path.dirname(zip_dir), 'cats_and_dogs_filtered')
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')
validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')

train_image_generator = ImageDataGenerator()
validation_image_generator = ImageDataGenerator()

train_data = train_image_generator.flow_from_directory(batch_size=2000,
                                                       directory=train_dir,
                                                       shuffle=True,
                                                       target_size=(IMG_SHAPE, IMG_SHAPE),
                                                       class_mode='binary').next()
val_data = validation_image_generator.flow_from_directory(batch_size=1000,
                                                          directory=validation_dir,
                                                          shuffle=False,
                                                          target_size=(IMG_SHAPE, IMG_SHAPE),
                                                          class_mode='binary').next()

cd_train_inputs, cd_train_labels = train_data
cd_test_inputs, cd_test_labels = val_data

print(cd_train_inputs.shape)
print(cd_train_labels.shape)
print(cd_test_inputs.shape)
print(cd_test_labels.shape)

index = np.random.randint(len(cd_train_inputs))
plt.imshow(cd_train_inputs[index] / 255)
plt.show()
print("Label:", cd_train_labels[index])

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same'))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), padding='same'))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), padding='same'))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512))
model.add(Dropout(0.7))
model.add(Activation('relu'))
model.add(Dense(2))
model.add(Activation('softmax'))

opt = keras.optimizers.legacy.RMSprop(learning_rate=0.0001, decay=1e-6)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

history = model.fit(cd_train_inputs, to_categorical(cd_train_labels),
                    validation_data=(cd_test_inputs, to_categorical(cd_test_labels)),
                    epochs=30)
plot_acc(history)
print(model.summary())

from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.utils import to_categorical
from keras.layers import Dense, Flatten
from keras.optimizers import SGD

train_data = train_image_generator.flow_from_directory(batch_size=2000,
                                                       directory=train_dir,
                                                       shuffle=True,
                                                       target_size=(224, 224),
                                                       class_mode='binary').next()
val_data = validation_image_generator.flow_from_directory(batch_size=1000,
                                                          directory=validation_dir,
                                                          shuffle=False,
                                                          target_size=(224, 224),
                                                          class_mode='binary').next()

cd_train_inputs, cd_train_labels = train_data
cd_test_inputs, cd_test_labels = val_data

model = VGG16(include_top=False, input_shape=(224, 224, 3))

for layer in model.layers:
    layer.trainable = False

flat1 = Flatten()(model.layers[-1].output)
class1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)
output = Dense(2, activation='sigmoid')(class1)

model = Model(inputs=model.inputs, outputs=output)

opt = SGD(learning_rate=0.001, momentum=0.9)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(cd_train_inputs, to_categorical(cd_train_labels),
                    validation_data=(cd_test_inputs, to_categorical(cd_test_labels)),
                    epochs=2)
plot_acc(history)

print(model.summary())
