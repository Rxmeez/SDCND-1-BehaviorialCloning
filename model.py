import csv
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Convolution2D
import sklearn
from sklearn.model_selection import train_test_split


def crop_resize(image):
    """
    Crop the images horizon and car bonnet
    and resize image (64, 64)
    """
    shape = image.shape
    image = image[math.floor(shape[0]/3):shape[0]-25, 0:shape[1]]
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


def aug_brightness(image):
    """
    Adds random value of brightness to the image
    """
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    image = np.array(image, dtype=np.float64)
    random_bright = .5+np.random.uniform()
    image[:, :, 2] = image[:, :, 2] * random_bright
    image[:, :, 2][image[:, :, 2] > 255] = 255
    image = np.array(image, dtype=np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
    return image


def add_random_shadow(image):
    """
    Applies random shadows to the image
    """
    top_y = 320 * np.random.uniform()
    top_x = 0
    bot_x = 160
    bot_y = 320 * np.random.uniform()
    image_hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    shadow_mask = 0 * image_hls[:, :, 1]
    X_m = np.mgrid[0:image.shape[0], 0:image.shape[1]][0]
    Y_m = np.mgrid[0:image.shape[0], 0:image.shape[1]][1]
    shadow_mask[((X_m-top_x)*(bot_y-top_y)-(bot_x-top_x)*(Y_m-top_y) >= 0)] = 1
    if np.random.randint(2) == 1:
        random_bright = .5
        cond1 = shadow_mask == 1
        cond0 = shadow_mask == 0
        if np.random.randint(2) == 1:
            image_hls[:, :, 1][cond1] = image_hls[:, :, 1][cond1]*random_bright
        else:
            image_hls[:, :, 1][cond0] = image_hls[:, :, 1][cond0]*random_bright
    image = cv2.cvtColor(image_hls, cv2.COLOR_HLS2RGB)
    return image


def preprocess(image, angle):
    """
    Add all the preprocess steps into one function
    """
    rand_flip = np.random.randint(2)  # 0 or 1
    image = aug_brightness(image)
    image = add_random_shadow(image)
    if rand_flip == 1:
        image, angle = flip_vertical(image, angle)
    image = crop_resize(image)
    return image, angle


# Samples where all the data points will be stored.
samples = []
# Reading the data points from the file
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        # Removes data where the steering wheel is less than 0.15 and leaves 10% of data points.
        if abs(float(line[3])) < 0.15 and np.random.uniform() < 0.9:
            continue
        samples.append(line)
# Data split of samples for test and validation
train_samples, validation_samples = train_test_split(samples, test_size=0.2)


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
            sklearn.utils.shuffle(samples)
            for offset in range(0, num_samples, batch_size):
                batch_samples = samples[offset:offset + batch_size]

                images = []  # Where images are stored
                measurements = []  # Where steering data is stored
                for batch_sample in batch_samples:
                    for i in range(3):
                        # Source for image (center[0], left[1], right[2])
                        filename = batch_sample[i].split('/')[-1]
                        current_path = "./data/IMG/{}".format(filename)
                        # Getting steering correction for images left and right
                        correction = 0.25  # Self-assigned value
                        steering = float(batch_sample[3])
                        if i == 1:   # Left Camera
                            steering = steering + correction
                        elif i == 2:  # Right Camera
                            steering = steering - correction
                        else:  # Center Camera
                            steering = steering
                        # Read image
                        img = cv2.imread(current_path)
                        # BGR2RGB because drive.py reads RGB
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        # Image preprocess
                        img, steering = preprocess(img, steering)
                        # Append all the images and measurements
                        images.append(img)
                        measurements.append(steering)

                X_train = np.array(images)
                y_train = np.array(measurements)
                # Shuffle data
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
model.add(Dropout(0.5))
model.add(Convolution2D(64, 3, 3, activation='elu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100, activation='elu'))
model.add(Dropout(0.5))
model.add(Dense(50, activation='elu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='elu'))
model.add(Dense(1))  # Output

model.compile(loss='mse', optimizer='adam')
# 30000 seem like enought data to train on
history_object = model.fit_generator(train_generator, samples_per_epoch=30000, validation_data=validation_generator, nb_val_samples=6000, nb_epoch=10, verbose=1)

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

# Save model to model.h5 to later be used by drive.py
print('Saving model...')
model.save('model.h5')
print('model.h5 Saved')
