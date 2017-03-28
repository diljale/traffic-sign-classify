#**Traffic Sign Recognition** 
---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image4]: ./data/20.jpg "Speed limit 20"
[image5]: ./data/30.jpg "Speed limit 30"
[image6]: ./data/80.jpg "Speed limit 80"
[image7]: ./data/ahead.jpg "Ahead only"
[image8]: ./data/left.jpg "Keep left"

## Rubric Points
---

You're reading it! and here is a link to my [project code](https://github.com/diljale/traffic-sign-classify)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is [32, 32]
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third code cell of the IPython notebook.  

Here is an exploratory visualization of an image in the data set.

![alt text][image2]

###Design and Test a Model Architecture

####1. Pre-processing

The code for this step is contained in the fourth code cell of the IPython notebook.

I converted image from RGB to grayscale and then normalized it to make sure input data is between 0.1 and 0.9, making input space with zero mean and small variance
I also split training set into training and validation set

####2. Final architecture

The code for my final model is located in the seventh cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 normalized grayscale image   			| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 28x28x6 	|
| RELU					|											   |	
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				    |
| Convolution 5x5     	| 1x1 stride, same padding, outputs 10x10x16 	 |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 					|
| Fully connected		| Input 400, Output 240 .        				|
| Fully connected		| Input 240, Output 120 .        				|
| Fully connected		| Input 120, Output 84 .        				|
| Fully connected		| Input 84, Output 43 .        				    |
| Softmax				|          									| 


####3. Describe how, and identify where in your code, you trained your model.

The code for training the model is located in the eigth cell of the ipython notebook. 

To train the model, I used an AdamOptimizer with batch size 256, number of epochs 30 and learning rate of 0.001

It gives accuracy of 0.967 on validation set and 0.860 on testing set
 
####4. Approach
I started with LeNet architecture and found that i didn't get good accuracy on validation set and test set. I then started changing the design by adding some layers like changing convolution layer to 3x3 or adding another convolution layer, but realized that this is incorrect approach. Correct approach would be to visualize each layer's output to contruct the network. I didn't have enough time to implement this yet, so i had to make some guesses as to what could improve accuracy of the archtiecture. By changing the layers I finally found a design which gave decent results on the dataset, but as noted below failed to generalize to new data.

The reason i started with LeNet architecture was because it is built to classify images into known labels. I knew that LeNet would not be sufficient to classify Traffic signs as Traffic signs are more complicated than alphabets since it has complex shapes in it. 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web. These are difficult as they have much higher fidelity and artificial background which is very different from training set:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

####2. Discuss the model's predictions on these new traffic signs .

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. compared of 86.7% on testing set.

The top 5 softmax probabilities are:
TopKV2(values=array([[  9.99984384e-01,   7.77230071e-06,   5.33014463e-06,
          2.27935175e-06,   1.44428526e-07],
       [  9.99999881e-01,   6.00122263e-08,   2.11380954e-10,
          2.18747347e-11,   1.06643676e-12],
       [  9.92061138e-01,   7.30022555e-03,   6.37910911e-04,
          3.77597871e-07,   2.77417882e-07],
       [  9.94185507e-01,   5.81437722e-03,   1.31712497e-07,
          1.70050747e-08,   7.33186123e-10],
       [  1.00000000e+00,   8.06565748e-09,   2.21623414e-10,
          8.60244376e-11,   6.34172229e-12]], dtype=float32),
indices=array([[ 5, 42,  7,  3,  2],
       [39, 37, 33,  4, 13],
       [ 1,  0, 40, 21,  6],
       [16,  3, 28, 35, 11],
       [ 1,  6, 12, 32,  0]], dtype=int32))
       
This shows that my model doesn't do too well on images from web.  Unfortunately i  didn't get time to improve my model.
Some ideas to improve the model:
1] Add more data
2] Visualize each CNN layer to understand shortcomings of training
3] Try adding more layers 
