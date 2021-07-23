### The Concept of YOLO

* A model for Object Detection. It is very fast, and able to run in real-time. It has 3 versions.
* YOLO divides the input image into an SXS grid. Each grid cell predicts only one object.
* Lets look at this on a small scale. Divide this image into 3x3 cells (9 Grid Cells) and assign the center of the object to that grid cell. This grid cell is responsible to predict the object.

![Decide Grid](https://github.com/joyjeni/mlguides/blob/master/Yolov3/images/1.png)

### Center of the object grid Predicts object

Each grid cell has 8 objects. In Yolo we predict a feature map of 3 dimensions. The size of feature map is image width x image height x8


![each Grid](https://github.com/joyjeni/mlguides/blob/master/Yolov3/images/2.png)
           
For those grid cells with no object detected, it's pc=0 and we don't care about the rest of the other values. Thats what "?" means in the graph.

### Bounding boxes

* bx: The center of the object according to the x coordinate, and has value ranging from 0-1
* by: The center of the object according to the y coordinate, and has a value ranging from 0-1
* bh: height of the boinding box, the value could be greater than 1.
* bw: width of the bounding box, the value could be greater than 1.

![each Grid](https://github.com/joyjeni/mlguides/blob/master/Yolov3/images/3.png)

### Anchor Boxes
1. There is a limitation with only having grid cells.
2. Say we have multiple objects in the same grid cell.For instance, there's a person standing in front of a car and their bounding box centers are so close.

Faster RCNN use 9 anchor box no grid cells. It uses region proposal.

![anchor_box](https://github.com/joyjeni/mlguides/blob/master/Yolov3/images/4.png)


![anchor_box](https://github.com/joyjeni/mlguides/blob/master/Yolov3/images/5.png)

* Each grid cell now has two anchor boxes, where each anchor box acts like a container. Meaning now each cell can predict up to 2 objects.
* Choosing anchor boxes with two different shapes is because when we make a cecision as to which object is put in which anchor box, we look at their shapes, nothing how similar one objects bounding box shape is to the shape of the anchor box. For the above exampple, the person will be associated with the tall anchor box since their shape is more similar.

* As a result, the output of one grid cel will be extended to contain information for two anchor boxes.
* For example, the center grid cell in the image now has 8x2 output labels in total, as shown below. 3x3x16

The General Formula:
(NXN)x[num_anchors x (5+num_classes)]

Here,
N=3
num_anchors=2,
num_casses=3
(3x3)x[2x(5+3)]

![anchor_box](https://github.com/joyjeni/mlguides/blob/master/Yolov3/images/6.png)

### Evaluation Metric

* Instead of defining a box by its center point, width and height, lets define it using its two corners. Upper left and lower right.(x1,y1,x2,y2)
* To compute the intersection of two boxes, we start off by finding the intersection areas of two corners.

![eval_metric](https://github.com/joyjeni/mlguides/blob/master/Yolov3/images/7.png)

![eval_metric](https://github.com/joyjeni/mlguides/blob/master/Yolov3/images/8.png)

### Non max supression
* Non- max suppression is a common algorithm used for cleaning up when multiple grid cells are predicted the same object.
* For the example below, the model outputs three predictions for the truck in the center. There are three bounding boxes, but we only need one. The thicker the predicted bounding box, the more confident the prediction is that means a higher pc value.

***Algorithm***
1. Discard all boxes with pc less or equal to 0.6
2. Pick the box with the largest pc output as a prediction.
3. Discard any remaining box with IoU greater than or equal to 0.5

### 1.  Discard all boxes with pc (Confidence)11 less or equal to 0.6
1. Given an image, the YOLO model will generate an output matrix of shape (3,3,2,8) which means each of the grid cells will have two predictions, even for those grid cells that dont have any object inside.
2. Now we need to filter with a threshold by "class scores"
3. The class scores are computed by multiplying pc with the individual class output(c1,c2,c3)
![nms](https://github.com/joyjeni/mlguides/blob/master/Yolov3/images/9.png)

![nms](https://github.com/joyjeni/mlguides/blob/master/Yolov3/images/10.png)
### YOLO in Depth
* YOLO makes use of only convolutional layers, making it a fully convolutional network(FCN). It has layers with skip connections and upsampling layers. No form of pooling is used, and a convolutional layer with stride 2 is used to downsample the feature maps. This helps in preventing loss of low-level features often 
attributed to pooling.

* The network downsamples the image by a factor called the stride of the network. For example, if the stride of the network is 32, then an input image of size 416x416 will yield an output of size 13x13. Generally, ***stride of any layer in the network is equal to the factor by which the output of the layer is smaller than the input image to the network.***

*** In YOLO, the prediction is done by using a convolutional layer which uses 1x1 convolutions, and the output is a feature map. Depth-wise, we have (Bx(5+C)) entries in the feature map. We expect each cell of the feature map to predict an object through one of its bounding boxes if the center of the object falls in the respective field of that cell.***

YOLO
* num_anchors(bounding boxes)=3 for each scale
* We have 3 scales
* We have 80 classes

![yolo output](https://github.com/joyjeni/mlguides/blob/master/Yolov3/images/11.png)


* Each grid cell now has two anchor boxes, where each anchor box acts like a container. Meaning now each grid cell can predict upto 2 objects.
* Choosing anchor boxes with two different shapes is because when we make a decision as to which object is put in which anchor box, we look at their shapes, nothing how similar one objects bounding box shape is to the shape of the anchor box. For the above example, the person will be associated with the tall anchor box since their shape is more similar.
* As a result, the output of one grid cell will be extended to contain information for two anchor boxes.
* For example, the center grid cell in the image now has 8x2 output labels in total, as shown below.

![yolo output](https://github.com/joyjeni/mlguides/blob/master/Yolov3/images/12.png)

### Predictions across different scales
* Yolov3 makes prediction across 3 different scales. The detection layer is used make detection at feature maps of three different sizes, having strides 32,16,8 respectively. This means with an input of 416x416, we make detections on scales.

13x13, 26x26 and 52x52

* The network downsamples the input image until the first setection layer, where a detection is made using feature maps of a layer with stride 32. Further, layers are upsampled by a factor of 2 and concatenated with feature maps of a previous layers having identical feature map sizes. Another detection is now made at layer with stride 16. The same upsampling procedure is repeated, and a final detection is made at the layer of stride 8.


![yolo output](https://github.com/joyjeni/mlguides/blob/master/Yolov3/images/13.png)

### Yolo V3 Architecture.
![yolo output](https://github.com/joyjeni/mlguides/blob/master/Yolov3/images/14.png)

Darknet 53 consists of 53 layers. and another 53 layers for detection. Totally 106 layers present.
![yolo arch](https://github.com/joyjeni/mlguides/blob/master/Yolov3/images/y_1_architecture.png)

### Stride

![yolo arch](https://github.com/joyjeni/mlguides/blob/master/Yolov3/images/y_2_stride.png)

### Networks Input

![network_input](https://github.com/joyjeni/mlguides/blob/master/Yolov3/images/y_3_network_input.png)

### Detection at 3 scales

82, 94,106

![detection](https://github.com/joyjeni/mlguides/blob/master/Yolov3/images/y_4_detections.png)
* Prediction at 3 scales help to predict small objects


![strides](https://github.com/joyjeni/mlguides/blob/master/Yolov3/images/y_5_strides.png)

### Feature maps

Yplo V3 has 3 feature maps.
(13x13), (26x26), (52x52)

Each feature map predicts 3 feature maps. They have same spatial dimensions. The shape of detection kernel also has its depth

### Number of bounding box attributes

\begin{equation}
b+(5*C) 
\end{equation}
b - Number of bounding box
C- Nimber of classes



Example:

Input image is 416x416, and stride of the network is 32. The dimensions of the feature map will be 13x13. We then divide the input image into 13x13 cells.
![yolo output](https://github.com/joyjeni/mlguides/blob/master/Yolov3/images/15.png)

### Prediction of the Anchor Boxes or Priors
* It might make sense to predict the width and height of the bounding box, but in practice, that leads to unstable gradients during training. Instead, most of the modern object detectors offsets(*** which is how much we should we move the predicted bounding box in order to get the desired bounding box) to pre-defined default bounding boxes(anchors)
* Then, these transforms are applied to the anchor boxes to obtain the prediction. YOLO v3 has three anchors, which result in prediction of three bounding boxes per cell.


![yolo output](https://github.com/joyjeni/mlguides/blob/master/Yolov3/images/16.png)


![yolo output](https://github.com/joyjeni/mlguides/blob/master/Yolov3/images/17.png)


![yolo output](https://github.com/joyjeni/mlguides/blob/master/Yolov3/images/18.png)

![yolo output](https://github.com/joyjeni/mlguides/blob/master/Yolov3/images/19.png)

### Why passing through Sigmoid

For example, consider the case of our dog image. If the prediction for center is (0.4,0.7), then this means that the center liesat
(6.4,6.7) on the 13x13 feature map. Since the top left coordinates of the red cell are (6,6)

* But wait, what happens if the predicted x,y co-ordinates are greater than one, say (1.2,0.7). This means center lies at (7.2,6.7)
Notice the center now lies in cell just right to our red cell, or the 8th cell un the 7th row. This breaks theory behind YOLO because
if we postulate that the red box is responsible for predicating the dog, the center of the dog must lie in the red cell, and not in the one beside it.

* Therefore, to remedy this problem, the output is passed through a sigmoud function, which squashes the output in a range from 0 to 1, effectively keeping the center in the grid which is predicting.


![yolo output](https://github.com/joyjeni/mlguides/blob/master/Yolov3/images/20.png)

### Class confidence
* Class confidences represents the probabilities of the detected object belonging to a particular cass (Dog, Cat, banana, car etc). Before V3 YOLO used to softmax the class scores.

* However, that design choice has been dropped in V3, and authors used sigmoid instead. The reason is that softmax class scores assume that the classes are mutually exclusive. In simple words, if an object belongs to one class, then its guaranteed it cannot belong to another class.
* Each class score is predicted using logistic regression and a threshold is used to predict multiple labels for an object. Classes with scores higher than threshold are assigned to the box.

#### Test your understanding

1. What are the total numbers of bounding boxes that YOLO predicts?

(13x13x3)+(26x26x3) + (52x52x3) = 10,647

2. Why are we predicting for 3 different scales?
This helps YOLO V3 get better at detecting small objects.

### Thresholding by Object Confidence

First, we filter boxes based on their objectness score. Generally, boxes having scores below a threshold are ignored.

### Choice of anchor boxes

* YOLO V3, in total uses 9 anchor boxes. Three for each scale. It also uses K-Means clustering to generate 9 anchors.
* The anchors are then arranged in descending order of a dimension. The three biggest anchors are assigned for the first scae, the next three for the second scale, and the last three for the third.
* At each scale, every grid can predict 3 boxes using 3 anchors. There are 3 scales.

### Yolo V2 loss function

![yolo_loss](https://github.com/joyjeni/mlguides/blob/master/Yolov3/images/21.png)

More detail
![yolo_loss](https://github.com/joyjeni/mlguides/blob/master/Yolov3/images/22.png)

* To compute the loss for the true positive, we only want one of them to be responsible for the object. For this purpose, we select the one with the highest IoU with the ground truth.

The YOLO loss function is composed of:

* The classification loss
* The ocalization loss (errors between the predicted boundary box and the ground truth)
* The confidence loss (The objectness of the box)

### Classification Loss

If an object is detected, the classification loss at each cell is the squared error of the class conditional probabilities for each class.
![yolo_loss](https://github.com/joyjeni/mlguides/blob/master/Yolov3/images/23.png)

### Localization Loss

The localization loss measures the errors in the predicted boundary box locations and sizes. We only count the box responsible for detecting the object.

![yolo_loss](https://github.com/joyjeni/mlguides/blob/master/Yolov3/images/24.png)

### Confidence Loss

![yolo_loss](https://github.com/joyjeni/mlguides/blob/master/Yolov3/images/25.png)

Together we get

![yolo_loss](https://github.com/joyjeni/mlguides/blob/master/Yolov3/images/26.png)

It is opposite of localization loss. 
NB: How class imbalance is solved. by multiplying whole term by 0.5


###Note

1. In Yolo V2 there is no anchor box.
2. In Yolo 1,2 one prediction
3. In Yolo V3, squared errors replaced by binary cross-entrophy error terms.Object confidence and class predictions in YOLO v3 are now predicted through logistic regression.

4. While we are training the detector, for each ground truth box, we assign a bounding box, whose anchor has the maximum overlap with the ground truth box.


![yolo_loss](https://github.com/joyjeni/mlguides/blob/master/Yolov3/images/26.png)

### YOLO V3 Results

![yolo_loss](https://github.com/joyjeni/mlguides/blob/master/Yolov3/images/27.png)

#### Reference

1. https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/
2. https://towardsdatascience.com/yolo-v3-object-detection-53fb7d3bfe6b
3. https://heartbeat.fritz.ai/gentle-guide-on-how-yolo-object-localization-works-with-keras-part-2-65fe59ac12d
4. https://jonathan-hui.medium.com/real-time-object-detection-with-yolo-yolov2-28b1b93e2088
5. Udemy Course, Fawaz Sammani, "The Complete Neural Networks Bootcamp: Theory, Applications"
