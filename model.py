import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Activation, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

lines = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []
for line in lines:
    # Get image source path
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = './data/IMG/' + filename
    # Get steering angle for a line
    steering_center = float(line[3])
    # Adjusted steering measurements for the side camera images
    correction = 0.2
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

    '''
    # Get images
    image = cv2.imread(current_path)
    vertical_flip_image = cv2.flip(image, 1)
    # Append images
    images.append(image)
    images.append(vertical_flip_image)
    # Get meansurements
    measurement = float(line[3])
    vertical_measurement_flip = measurement * -1.0
    # Append measurements
    measurements.append(measurement)
    measurements.append(vertical_measurement_flip)
    '''
X_train = np.array(images)
y_train = np.array(measurements)

# print(X_train.shape, y_train.shape)

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0)), input_shape=(160, 320, 3)))
model.add(Convolution2D(6, 5, 5, activation='elu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(6, 5, 5, activation='elu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(120))
model.add(Activation('elu'))
model.add(Dense(84))
model.add(Activation('elu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=2, verbose=1)

print('Saving model...')
model.save('model.h5')
print('model.h5 Saved')
