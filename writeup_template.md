# **Behavioral Cloning**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/cnn-architecture-624x890.png "NVIDIA Model Visualization"
[image2]: ./examples/center-camera-lane.jpg "Center Vehicle Camera"
[image3]: ./examples/unbalance.png "Unbalanced Dataset"
[image4]: ./examples/balance.png "Balanced Dataset"
[image5]: ./examples/flip-images.png "Flipped Images Augmentation"
[image6]: ./examples/brightness.png "Random Brightness Augmentation"
[image7]: ./examples/shadows.png "Random Shadows Augmentation"
[image8]: ./examples/crop-64x64.png "Crop and Resize Image"
[image9]: ./examples/Figure_2.png "Model Mean Square Error Loss"

---

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network based on [NVIDIA's architecture](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/)

The model includes ELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer.

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting.

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, using right and left cameras to try and keep the car in the middle, as well as augment images for different senarios, and used balanced data distribution.

For details about how I created the training data, see the next section.

---

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to try, analyze failures and improve.

My first step was to use a convolution neural network model similar to the Yann Lecun's LeNet. I thought this model might be appropriate because its a good place to start with a small model and having implemented it before I had a good understanding of it.

In order to gauge how well the model was working, I split my image and steering angle data into training and validation set. Intially what I was looking for was that it should be able to drive relatively in the middle of the road and would not deviate on a straight road. This showed my that my model was working and was a good start. However when I was trying to go around the corner, it would fail, this could mean to add extra Convolution layer to pick up more features.

Following Udacity's suggestion I implemented NVIDIA's self-driving model architecture.

In my testing I found that there would be generally low mean squared error on the training set but a high mean squared error on the validation set, this implied that the model was overfitting. One of the ways I modified the model was to add dropout of 50% to generalise the model.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track around the first corner with basic LeNet architecture, it then struggled on the bridge where it would hug the edges, also I saw a pattern where it would deviate where ever there were shadows. One part of the track which gave the most issue was the 2nd corner near the dirt track. To improve the driving behavior in these cases, I moved the modified version of NVIDIA's model, balance the steering data distribution by removing data where the steering wheel was close to 0, and going over certain parts of the track multiple times.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes:
- Input Size: (64,64,3)
- Convolution: 5x5, filter: 24, strides: 2x2, activation: ELU
- Convolution: 5x5, filter: 36, strides: 2x2, activation: ELU
- Convolution: 5x5, filter: 48, strides: 2x2, activation: ELU
- Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
- Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
- Drop out (0.5)
- Fully connected: neurons: 100, activation: ELU
- Fully connected: neurons: 50, activation: ELU
- Fully connected: neurons: 10, activation: ELU
- Fully connected: neurons: 1 (Output)

Here is a visualization of the architecture:

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

I initially used Udacity dataset and then on top of that I recorded two laps on the track using center lane driving( i.e. stay in the middle of the lane as much as possible) as this is good driving behavior. Here is an example image of center lane driving:

![alt text][image2]

I found that my vehicle performanced poorly in certain parts of the track and to improve it I recorded that part multiple times. What this provided was more data values at the part, in my case it was the sharp corners, which were extremely low number of data point when we look at the unbalanced distribution:

![alt text][image3]

To balance the dataset, two methods were introduced recording the track at sharper corners, but that would take too long, and increase the size of our data. So, the second method was more effective were most datapoints at 0 steering were removed, this gave the balanced distribution:

![alt text][image4]

To augment the data sat, I also flipped images and angles thinking that this would not train the vehicle for only left turns as we go anti-clockwise around the track. For example, here is an image that has then been flipped:

![alt text][image5]

Other augmentation techniques that were used are random brightness level of the images and shadows on the images. This way it will learn to still stay on the track when it see a shadow and not deviate. Images below are example of brightness and shadow augmentation.

![alt text][image6]

![alt text][image7]

Another change that was made to the image was it was passed to the network in the size of 64 by 64, channel RGB and most of top and some of bottom of the image were cropped, environment and bonnet respectively.

![alt text][image8]

After I had my balanced data, I created a series of steps which my image will go through augmentation which were the following:

Preprocess steps
1. Brightness Augmentation
2. Shadow Augmentation
3. 50-50 Probability of Flipped Image
4. Crop and Resize Image

As I was using python generator I could if I wanted an infinity number of images, so the number of data collected did not matter.

A interesting bug I had in the system was my at one point my model was training on 3 images (left - center - right), which was being augmented 30000 times for 5 epoch and it managed to go around the first corner and the bridge but fail at the 2nd corner near the dirt track. This truly shows the power of augmenation.

The images were randomly suffled and were split into training and validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 10 as evidenced by graph below by being the lowest and in the autonomous mode it drove around the whole track without going off which is the real test. I used an adam optimizer so that manually training the learning rate wasn't necessary.

![alt text][image9]
