import os
import tensorflow as tf
from shutil import copyfile
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint

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