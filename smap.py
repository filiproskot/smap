import os
import tensorflow as tf
from shutil import copyfile
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
import cv2
import numpy as np

global IMG_HEIGHT
IMG_HEIGHT = 150
global IMG_WIDTH
IMG_WIDTH = 150

print('Number of testing images with mask: ', len(r'C:\Users\filip\PycharmProjects\Test\school\smap\data\with_mask'))
print('Number of testing images without mask:', len(r'C:\Users\filip\PycharmProjects\Test\school\smap\data\without_mask'))

epochs = int(input('How many epochs?\n')) #default 30


def trainTestSplit(source, trainPath, testPath, split_size):
    dataset = []

    for crnImage in os.listdir(source):
        data = source + '/' + crnImage
        if (os.path.getsize(data) > 0):
            dataset.append(crnImage)
    train_len = int(len(dataset) * split_size)
    train = dataset[0:train_len]
    test = dataset[train_len:len(dataset)]
    print('Train images with mask:', len(train))
    print('Test images without mask:', len(test))

    for trainDataPoint in train:
        crnTrainDataPath = source + '/' + trainDataPoint
        newTrainDataPath = trainPath + '/' + trainDataPoint
        copyfile(crnTrainDataPath, newTrainDataPath)

    for testDataPoint in test:
        crnTestDataPath = source + '/' + testDataPoint
        newTestDataPath = testPath + '/' + testDataPoint
        copyfile(crnTestDataPath, newTestDataPath)


def trainModel():
    training_dir = 'data/train'
    train_datagen = ImageDataGenerator(rescale=1.0 / 255,
                                       rotation_range=40,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True,
                                       fill_mode='nearest')

    train_generator = train_datagen.flow_from_directory(training_dir,
                                                        batch_size=10,
                                                        target_size=(IMG_WIDTH, IMG_HEIGHT))
    validation_dir = "data/test"
    validation_datagen = ImageDataGenerator(rescale=1.0 / 255)
    validation_generator = validation_datagen.flow_from_directory(validation_dir,
                                                                  batch_size=10,
                                                                  target_size=(IMG_WIDTH, IMG_HEIGHT))
    checkpoint = ModelCheckpoint('model-{epoch:03d}.model', monitor='val_loss', verbose=0, save_best_only=True,
                                 mode='auto')


model = tf.keras.models.Sequential([
    Conv2D(32, 3, activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, 3, activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, 3, padding='same', activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dropout(0.5), #0.3
    Dense(256, activation='relu'),
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

trainTestSplit('data/with_mask', 'data/train/training_with_mask', 'data/test/test_with_mask', 0.75)
trainTestSplit('data/without_mask', 'data/train/training_without_mask', 'data/test/test_without_mask', 0.75)
trainModel()

labels_dict = {0: 'no mask', 1: 'mask'}
color_dict = {0: (0, 0, 255), 1: (0, 255, 0)}

size = 4
webcam = cv2.VideoCapture(0)
classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:
    rval, im = webcam.read()
    im = cv2.flip(im, 1, 1)

    mini = cv2.resize(im, (im.shape[1] // size, im.shape[0] // size))
    faces = classifier.detectMultiScale(mini)

    for (x, y, w, h) in faces:
        face_img = im[y:(y + h) * size, x:(x + w) * size]
        resized = cv2.resize(face_img, (IMG_WIDTH, IMG_HEIGHT))
        normalized = resized / 255.0
        reshaped = np.reshape(normalized, (1, IMG_WIDTH, IMG_HEIGHT, 3))
        reshaped = np.vstack([reshaped])
        result = model.predict(reshaped)

        label = np.argmax(result, axis=1)[0]

        cv2.rectangle(im, (x * size, y * size), ((x + w) * size, (y + h) * size), color_dict[label], 2)
        cv2.rectangle(im, (x * size, (y * size) - 40), ((x + w) * size, y * size), color_dict[label], -1)
        cv2.putText(im, labels_dict[label], (x * size + 10, (y * size) - 10), cv2.FONT_HERSHEY_DUPLEX, 0.8,
                    (255, 255, 255), 2)

    cv2.imshow('Test', im)
    key = cv2.waitKey(10)
    if key == 27:
        break

webcam.release()
cv2.destroyAllWindows()