#**Behavioral Cloning Project**

The goals / steps of this project are the following:

* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./plots/centre_lane_driving.jpg "Centre Lane Driving"
[image2]: ./plots/left_camera.jpg "Left Camera"
[image3]: ./plots/centre_camera.jpg "Centre Camera"
[image4]: ./plots/right_camera.jpg "Right Camera"
[image5]: ./plots/original.jpg "Original Image"
[image6]: ./plots/flipped.jpg "Flipped Image"
[image7]: ./plots/grayscale.jpg "Grayscale Image"
[image8]: ./plots/loss.png "Loss"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---

###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:

* [model.py](model.py) containing the script to create and train the model
* [drive.py](drive.py) for driving the car in autonomous mode
* [model.h5](model.h5) containing a trained convolution neural network 
* [writeup_report.md](writeup_report.md) (this file) summarizing the results
* [run1.mp4](run1.mp4) showing the model being used to autonomously drive around track 1
* [run2.mp4](run2.mp4) showing the model being used to autonomously drive around track 2

####2. Submission includes functional code

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing ```python drive.py model.h5```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My final model is based on the Nvidea architecture consists of the following layers:

| Layer | Description | 
|:---:|:---:| 
| Input | 160x320x1 grayscale image |
| Cropping | Crops off the top 70 pixels and bottom 25 pixels of the image, output = 65x320x1 | 
| Convolution 5x5 | 2x2 stride, output = 31x158x24 output |
| RELU | activation function |
| Convolution 5x5 | 2x2 stride, output = 14x77x36 output |
| RELU | activation function |
| Convolution 5x5 | 2x2 stride, output = 5x37x48 |
| RELU | activation function |
| Convolution 5x5 | output = 3x35x64 |
| RELU | activation function |
| Convolution 5x5 | output = 1x33x64 |
| RELU | activation function |
| Flatten | flatten outputs to 1-dimensional set of inputs, output = 2112 |
| Dropout | keep probability of 0.8 |
| Fully connected	| output = 100 |
| Dropout | keep probability of 0.8 |
| Fully connected	| output = 50 |
| Dropout | keep probability of 0.8 |
| Fully connected	| output = 10 |
| Dropout | keep probability of 0.8 |
| Fully connected	| output = 1 |

defined in ```def nvidea_model()``` in model.py 

The model includes a cropping layer to exclude background scenery and the car bonnet from the input data so that the model focuses on the road features. RELU layers are used to introduce non-linearity to the model and the data is normalized using a Keras lambda layer.

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting. 

The model was trained and validated on different data sets to ensure that the model was not overfitting. This includes data from both tracks driving in both clockwise and anti-clockwise directions.

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I aimed to keep the vehicle in the centre of the road during each training run and then used images from the centre, right and left cameras with appropriate steer angle corrections to train the model to drive back into the centre of the road when off course.

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to start from an existing architecture and then make incremental improvements.

My first step was to use the LeNet architecture (similar to that used for the traffic sign classification project). I thought this model might be appropriate because it is used to identify features in images.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. While the model didn't appear to be overfitting it performed poorly on the simulator when it attempted to drive the vehicle autonomously.

I then moved over to the Nvidea architecture. This performed better in the simulator but I noticed that the model had a low mean squared error on the training set but a higher mean squared error on the validation set. This implied that the model was overfitting. 

This manifested during simulation in the model driving near the centre of the second track (over the dashed lines) but on the first track it was steering to one side of the track and driving over the lane markings. 

To combat the overfitting, I modified the model so that it included a number of dropout layers.

Then I retrained the network and adjusted the number of epochs to run so that the model wasn't overfitting.

At this point the vehicle was able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture is based on the NVidea architecture (see earlier for a full description of each layer).

####3. Creation of the Training Set & Training Process

To capture good driving behaviour, I first recorded a lap of track one using centre lane driving. Here is an example image of centre lane driving:

![alt text][image1]

I then recorded another lap of track one driving in the opposite direction.

Then I repeated this process on track two in order to get more data points.

I made use of the centre camera image with the original steer angle to teach the model centre lane driving. I then augmented the data set with the left and right camera images. For these images I used a steering correction of +/-0.3 to teach the model how to recover when it deviates from the centre of the road. For example, here are the left, centre and right images with steer angle corrections:

| Left (Steer Correction = -0.3) | Centre (Steer Correction = 0.0) | Right (Steer Correction = 0.3) |
| --- | --- | --- |
| ![alt text][image2] | ![alt text][image3] | ![alt text][image4] |

In addition to driving in the opposite direction, I also flipped images and angles and added these to the training data set. I thought that this would prevent the model from favouring steering in one direction over another and would help teach the model about the symmetry of the problem being solved. For example, here is an image that has then been flipped:

![alt text][image5]
![alt text][image6]

Having done this I found that the model was capable of driving around track one but failed to get around track two - particularly on the sharper corners. I added some new training data from these difficult corners to help train the model to deal with them.

After the collection process, I had 5216 data points (which each had 3 camera images and the flipped versions of each of these so 31296 in total). I then preprocessed this data by turning the images into grayscale. I found this helped especially when I was using a mixture of training data from the two tracks (track one being brown speckled surface and track two being black tarmac).

![alt text][image5]
![alt text][image7]

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. I settled on 8 epochs to train my model. When more epochs were used the validation loss stopped decreasing while the training loss kept decreasing (suggesting overfitting) as seen here:

![alt text][image8]

I used an adam optimizer so that manually training the learning rate wasn't necessary.
