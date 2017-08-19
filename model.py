import csv
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
import sklearn
from sklearn.model_selection import train_test_split

ch, row, col = 3, 160, 320  # Image format


def crop_resize(image):
    """
    Crop the images horizon and car bonnet
    and resize image (64, 64)
    """
    shape = image.shape
    image = image[math.floor(shape[0]/5):shape[0]-25, 0:shape[1]]
    image = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)
    return image


def flip_vertical(image, angle):
    """
    Flips images and angle on a vertical line
    i.e right(+1) becomes left(-1)
    """
    flip_image = cv2.flip(image, 1)
    flip_angle = angle * (-1)
    return flip_image, flip_angle


def img_to_YUV(img):
    """ Converted Image from BGR to RGB """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# TODO: Fix Generator to read all of the data which is being augmeneted aswell


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
            sklearn.utils.shuffle(samples)
            for offset in range(0, num_samples, batch_size):
                batch_samples = samples[offset:offset + batch_size]

                images = []
                measurements = []
                for batch_sample in batch_samples:
                    for i in range(3):
                        # Source for image (center[0], left[1], right[2])
                        filename = line[i].split('/')[-1]
                        current_path = "./data/IMG/{}".format(filename)
                        # Image Color filter
                        img = crop_resize(img_to_YUV(cv2.imread(current_path)))
                        # Getting steering correction for images left and right
                        correction = 0.25
                        steering = float(line[3])
                        if i == 1:
                            steering = steering + correction
                        elif i == 2:
                            steering = steering - correction
                        else:
                            steering = steering
                        # Augment a flip of the image
                        aug_img, aug_steering = flip_vertical(img, steering)
                        # Append all the images and measurements
                        images.extend((img, aug_img))
                        measurements.extend((steering, aug_steering))

                X_train = np.array(images)
                y_train = np.array(measurements)
                yield sklearn.utils.shuffle(X_train, y_train)


train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

# NVIDIA model used
# Image normalization to avoid saturation and make gradients work better.
# Convolution: 5x5, filter: 24, strides: 2x2, activation: ELU
# Convolution: 5x5, filter: 36, strides: 2x2, activation: ELU
# Convolution: 5x5, filter: 48, strides: 2x2, activation: ELU
# Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
# Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
# Drop out (0.5)
# Fully connected: neurons: 100, activation: ELU
# Fully connected: neurons: 50, activation: ELU
# Fully connected: neurons: 10, activation: ELU
# Fully connected: neurons: 1 (Output)

model = Sequential()
model.add(Lambda(lambda x: (x / 127.5) - 1.0, input_shape=(64, 64, 3)))
model.add(Convolution2D(24, 5, 5, activation='elu', subsample=(2, 2)))
model.add(Convolution2D(36, 5, 5, activation='elu', subsample=(2, 2)))
model.add(Convolution2D(48, 5, 5, activation='elu', subsample=(2, 2)))
model.add(Convolution2D(64, 3, 3, activation='elu'))
model.add(Convolution2D(64, 3, 3, activation='elu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100, activation='elu'))
model.add(Dense(50, activation='elu'))
model.add(Dense(10, activation='elu'))
model.add(Dense(1))  # Output

model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=5, verbose=1)

# Print the keys contained in the history object
print(history_object.history.keys())

# Plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()


print('Saving model...')
model.save('model.h5')
print('model.h5 Saved')
