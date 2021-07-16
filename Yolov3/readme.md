### The Concept of YOLO

* A model for Object Detection. It is very fast, and able to run in real-time. It has 3 versions.
* YOLO divides the input image into an SXS grid. Each grid cell predicts only one object.
* Lets look at this on a small scale. Divide this image into 3x3 cells (9 Grid Cells) and assign the center of the object to that grid cell. This grid cell is responsible to predict the object.

![Decide Grid](https://github.com/joyjeni/mlguides/tree/master/Yolov3/images/1.png?raw=true)




#### Reference
1. https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/
2. https://towardsdatascience.com/yolo-v3-object-detection-53fb7d3bfe6b
3. https://heartbeat.fritz.ai/gentle-guide-on-how-yolo-object-localization-works-with-keras-part-2-65fe59ac12d
4. https://jonathan-hui.medium.com/real-time-object-detection-with-yolo-yolov2-28b1b93e2088
