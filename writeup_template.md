**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/test_images/visualization.png "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/test_images/14.jpg "Traffic Sign 1"
[image5]: ./examples/test_images/17.jpg "Traffic Sign 2"
[image6]: ./examples/test_images/24.jpg "Traffic Sign 3"
[image7]: ./examples/test_images/37.jpg "Traffic Sign 4"
[image8]: ./examples/test_images/40.jpg "Traffic Sign 5"

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

Here is a link to my [project code](https://github.com/hshirazi/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

## Data Set Summary & Exploration

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is [26 25]
* The number of unique classes/labels in the data set is 43

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distribted among 43 labels (traffic signs).

![alt text][image1]

## Design and Test a Model Architecture

**Pre-Prossesing:**

As a first step, I decided to convert the images to grayscale because when I used the LeNet model withought grayscaling, it was overfitting. Grayscaling reduces the number of channels from 3 to 1, hence less complexity and less chance of overfitting. It helped greatly. This also means color was not an important feature.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data because it reduces the range of values the network works with from [0,256] to [-1,1]. This a more controlled range and also makes overfitting less likely.


My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	2x2      	| 2x2 stride, valid padding, outputs 14x14x6 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16      									|
| RELU					|												|
| Max pooling	2x2      	| 2x2 stride, valid padding, outputs 5x5x16 				|
| Dropout					|					0.5 keep probability							|
| Fully connected		| 400 flatten input, outputs 84       									|
| Softmax				|        									|
 

To train the model, I used an AdamOptimizer with batch size of 128, 10 epochs, and learning rates of 0.01,0.01,0.005,0.005,0.003,0.003,0.001,0.001,0.001,0.001 for epochs 1,2,...,10.

My final model results were:
* training set accuracy of 0.994
* validation set accuracy of 0.951 
* test set accuracy of 0.936

I started with the LeNet architecture. It had a high accuracy on Training set but low accuracy on Validation set. I concluded that it was overfitting. I decided to simplify the model to solve the overfitting problem. I removed the last two fully connected layers. It was still overfitting. Then I tried dropout on the last layer (fully connected), so that features do not rely on each other too much. It was a great improvement. I was getting very good improvement on accuracy in the first few epochs, but it was getting worse in the later epochs. I concluded over shooting of the minimum cost was happening in the later epochs. So I decreased the learning rate for the later epochs. This improved the model accuracy for later epochs.

I started with LeNet architecture because it was successful in image recognition. By using convolutional layers, it solved the problem of image transition.
 
## Test the Model on New Images

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The images might be difficult to classify because of the different lighting on them and the shaddow of other images on them.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop Sign   									| 
| No Entry     			| No Entry 										|
| Road narrows on the right	| Road narrows on the right	|
| Go straight or left	      		| Go straight or left					 				|
| Roundabout mandatory			| Roundabout mandatory     							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 0.936.

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is almost certain that this is a Stop sign (probability of 0.9999), and the image does contain a Stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.9999         			| Stop sign   									| 
| 0.0000     				| Speed limit (120 km/h)	 										|
| 0.0000					| Turn right ahead											|
| 0.0000	      			| No Entry					 				|
| 0.0000				    | Give way    							|

For the second image, the model is almost certain that this is a No Entry (probability of 1.0), and the image does contain a No Entry. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0000         			| No Entry   									| 
| 0.0000     				| Stop Sign 										|
| 0.0000					| Keep right											|
| 0.0000	      			| Speed limit (30 km/h)					 				|
| 0.0000				    | No overtaking     							|

For the third image, the model is absolutely sure that this is a Road narrows on the right (probability of 0.9686), and the image does contain a Road narrows on the right.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.9686         			| Road narrows on the right  									| 
| 0.0302     				| Pedestrians 										|
| 0.0006					| Road ahead freezes easily and is then slippery											|
| 0.0003	      			| Bicycles crossing					 				|
| 0.0001				    | Right of way at the next crossroads    							|


For the fourth image, the model is almost certain that this is a Go straight or left (probability of 1.0000), and the image does contain a Go straight or left. The top five soft max probabilities were


| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0000         			| Go straight or left	   									| 
| 0.0000     				| Speed limit (30 km/h) 										|
| 0.0000					| Speed limit (70 km/h)											|
| 0.0000	      			| Speed limit (20 km/h)					 				|
| 0.0000				    | Ahead only     							|

For the fifth image, the model is almost certain that this is a Roundabout mandatory (probability of 0.915), and the image does contain a Roundabout mandatory. The top five soft max probabilities were


| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.9156         			| Roundabout mandatory	   									| 
| 0.0663     				| priority road 										|
| 0.0119					| Speed limit (100 km/h)											|
| 0.0048	      			| Speed limit (30 km/h)					 				|
| 0.0005				    | Speed limit (120 km/h)    							|


