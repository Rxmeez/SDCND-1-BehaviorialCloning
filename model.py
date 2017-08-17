import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
import sklearn
from sklearn.model_selection import train_test_split

ch, row, col = 3, 160, 320  # Image format

samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
            sklearn.utils.shuffle(samples)
            for offset in range(0, num_samples, batch_size):
                batch_samples = samples[offset:offset + batch_size]

                images = []
                measurements = []
                for batch_sample in batch_samples:
                    # Get image source path
                    source_path = line[0]
                    filename = source_path.split('/')[-1]
                    current_path = './data/IMG/' + filename
                    # Get steering angle for a line
                    steering_center = float(line[3])
                    # Adjusted steering measurements for the side camera images
                    correction = 0.3
                    steering_left = steering_center + correction
                    steering_right = steering_center - correction
                    # Read in images from center, left and right camera
                    center_source_path = line[0]
                    center_filename = source_path.split('/')[-1]
                    center_current_path = './data/IMG/' + filename
                    left_source_path = line[1]
                    left_filename = source_path.split('/')[-1]
                    left_current_path = './data/IMG/' + filename
                    right_source_path = line[2]
                    right_filename = source_path.split('/')[-1]
                    right_current_path = './data/IMG/' + filename
                    img_center = cv2.imread(center_current_path)
                    img_left = cv2.imread(left_current_path)
                    img_right = cv2.imread(right_current_path)
                    images.extend((img_center, img_left, img_right))
                    measurements.extend((steering_center, steering_left, steering_right))
                    augmented_img_center = cv2.flip(img_center, 1)
                    augmented_img_left = cv2.flip(img_left, 1)
                    augmented_img_right = cv2.flip(img_right, 1)
                    augmented_steering_center = steering_center * (-1)
                    augmented_steering_left = steering_left * (-1)
                    augmented_steering_right = steering_right * (-1)
                    images.extend((augmented_img_center, augmented_img_left, augmented_img_right))
                    measurements.extend((augmented_steering_center, augmented_steering_left, augmented_steering_right))

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
model.add(Lambda(lambda x: (x / 127.5) - 1.0, input_shape=(row, col, ch), output_shape=(row, col, ch)))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))
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
model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=3)

print('Saving model...')
model.save('model.h5')
print('model.h5 Saved')
