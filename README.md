# **Project 5 - Vehicle Detection**

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./writeup_images/1_vehicles_non_vehicles_training_dataset_examples.png "Vehicles and non-vehicles examples"
[image2]: ./writeup_images/2_spatial_binning_example.png "Spatial binning example"
[image3]: ./writeup_images/3_color_historgam_example.png "Color historgam example"
[image4]: ./writeup_images/4_hog_example.png "Hog example"
[image5]: ./writeup_images/5_sliding_windows.png "Sliding Windows"
[image6]: ./writeup_images/6_sliding_windows_vehicles_detected.png "Sliding windows vehicles detected"
[image7]: ./writeup_images/7_hog_sub_sampling_windows_detected_1.png "Hog sub-sampling windows detected 1"
[image8]: ./writeup_images/8_hog_sub_sampling_windows_detected_2.png "Hog sub-sampling windows detected 2"
[image9]: ./writeup_images/9_hog_sub_sampling_vehicles_detected_1.png "Hog sub-sampling vehicles detected 1"
[image10]: ./writeup_images/10_hog_sub_sampling_vehicles_detected_2.png "Hog sub-sampling vehicles detected 2"
[image11]: ./writeup_images/11_hog_sub_sampling_RGB.png "Hog sub-sampling RGB"
[image12]: ./writeup_images/12_hog_sub_sampling_HSV_vehicles_detected_1.png "Hog sub-sampling HSV vehicles detected 1"
[image13]: ./writeup_images/13_hog_sub_sampling_HSV_vehicles_detected_2.png "Hog sub-sampling HSV vehicles detected 2"
[image14]: ./writeup_images/14_feature_extraction_spatial_hist_hog_windows_detected.png "Feature extraction spatial hist hog windows detected"
[image15]: ./writeup_images/15_feature_extraction_spatial_hist_hog_vehicles_detected.png "Feature extraction spatial hist hog vehicles detected"
[image16]: ./writeup_images/16_combined_sliding_window_hog_sub_sampling_windows_detected.png "Combined sliding window hog sub-sampling windows detected"
[image17]: ./writeup_images/17_combined_sliding_window_hog_sub_sampling_windows_heatmap.png "Combined sliding window hog sub-sampling windows heatmap"
[image18]: ./writeup_images/18_combined_sliding_window_hog_sub_sampling_windows_heatmap_threshold.png "Combined sliding window hog sub-sampling windows heatmap threshold"
[image19]: ./writeup_images/19_combined_sliding_window_hog_sub_sampling_windows_vehicles_detected.png "Combined sliding window hog sub-sampling windows vehicles detected"


## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You are reading the Writeup / README for the Vehicle Detection project and here is a link to my project code [`P5_VehicleDetection.ipynb`](https://github.com/viralzaveri12/CarND-P5-VehicleDetection/blob/master/P5_VehicleDetection.ipynb)

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

In **Step 1** of IPython notebook `P5_VehicleDetection.ipynb`, I load all the `vehicle` and `non-vehicle` images. Below image shows few random examples of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

In **Section c) Histogram of Oriented Gradients (HOG) Features** of **Step 2 - Feature Extraction from Vehicle and Non-Vehicle Images** of IPython notebook `P5_VehicleDetection.ipynb`, the `get_hog_features` function is used to explore different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`). I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image4]

#### 2. Explain how you settled on your final choice of HOG parameters.

Deciding on the HOG parameters was an iterative process. The selection criteria for the final choice of HOG parameters was based on accuracy of the SVM classifier. Finally, HOG parameters I chose for training the classifier are:

```python
color_space = 'YUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 11 # HOG orientations
pix_per_cell = 16 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
```

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

In **Section a) Spatial Binning of Color Features** of **Step 2 - Feature Extraction from Vehicle and Non-Vehicle Images** of IPython notebook `P5_VehicleDetection.ipynb`, I explored spatial binning of color features.

![alt text][image2]

In **Section b) Color Histogram Features** of **Step 2 - Feature Extraction from Vehicle and Non-Vehicle Images** of IPython notebook `P5_VehicleDetection.ipynb`, I explored color histogram features.

![alt text][image3]


In **Section e) Extract Features from a list of Vehicle and Non-Vehicle Images** of **Step 2 - Feature Extraction from Vehicle and Non-Vehicle Images** of IPython notebook `P5_VehicleDetection.ipynb`, I extracted color and HOG features using the parameters mentioned in the above question from the training sets for both `vehicles` and `non-vehicles` classes.

In **Step 3 - Train the Classifier** of IPython notebook `P5_VehicleDetection.ipynb`, I use the support vector machine (SVM) as a classifier that works well with Hog features.

**With Color and HOG features:**
<pre>
Feature vector length           - 5568
Time train SVM Classifier       - 15.71 s
Test Accuracy of SVM Classifier - 99.13 %
Time to predict                 - 0.043 s
</pre>

Result of initial iterations with both **Color and HOG features:**

![alt text][image14]

With both **Color and HOG features**, I could not get rid of false detections (selecting larger threshold for heatmap would also reject the correct detections in few cases) as shown below:

![alt text][image15]

**With Only HOG features:**  
<pre>
Feature vector length           - 1188
Time train SVM Classifier       - 1.58 s
Test Accuracy of SVM Classifier - 98.31 %
Time to predict                 - 0.025 s
</pre>

From the initial iterations with **Only HOG features**, I could get rid of false detections while retaining all correct detections. So, I decided to extract only the HOG features for training the SVM classifier.

Below is the table listing the SVM classifier attributes. For all the color spaces, decision factors like time to train, accuracy, and time to predict were all very similar and indicated no clear demarcation for any particular choice.

| Color Space | Time to Train SVM Classifier (s) | Test Accuracy of SVM Classifier | Time to Predict (s) |
|:-----------:|:--------------------------------:|:-------------------------------:|:-------------------:|
| RGB         | 2.87  | 96.88 %  | 0.047  |
| HSV         | 1.24  | 98.23 %  | 0.045  |
| LUV         | 1.87  | 97.38 %  | 0.029  |
| HLS         | 1.58  | 97.66 %  | 0.031  |
| YUV         | 1.45  | 98.45 %  | 0.032  |
| YCrCb       | 1.49  | 98.14 %  | 0.072  |

Therefore, I trained the classifier with each color space listed above and observed the output.

**``With Color Space - RGB:``**

![alt text][image11]

With RGB color space, barely any detected window falls on the vehicles. So, RGB is ruled out.

**``With Color Space - HSV:``**

![alt text][image12]

![alt text][image13]

With HSV color space, correct vehicle detections also get eliminated while getting rid of false detections. So, HSV is ruled out.

**`Color Spaces - LUV, HLS, YUV, and YCrCb:`**

Between LUV, HLS, YUV, and YCrCb, the output of all these color spaces were fairly similar. All had no false detections and correct vehicle detections.

I trained the classifier with all LUV, HLS, YUV, and YCrCb color spaces and finally tested the classifier for each color space on the `project_video.mp4`

Color space **YUV** gave the best result for vehicle detection `project_video.mp4`

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

In **Step 5 -  Apply Functions on Test Images for Vehicle Detection** of IPython notebook `P5_VehicleDetection.ipynb`, I first tried **Section 1 - Sliding Window Implementation** that contains separate functions to determine number of windows, extract features, and sliding window search to make predictions.

![alt text][image5]

The below image shows multiple detections for single vehicle.

![alt text][image6]

Also, this implementation takes a long time (no metrics are recorded) to run on test images, leave alone processing on the videos. Thus, moving to a more efficient method for the sliding window approach is to extract the Hog features only once for each of a small set of predetermined window sizes (defined by a scale argument), and then can be sub-sampled to get all of its overlaying windows.

In **Step 5 - Apply Functions on Test Images for Vehicle Detection** of IPython notebook `P5_VehicleDetection.ipynb`, I tried **Section 2) a) Hog Sub-Sampling Window Search** with fixed parameters as below:
```python
ystart = 400
ystop = 656
scale = 1.5
```
Below are the examples of windows detected after Hog Sub-Sampling Window Search:

![alt text][image7]

![alt text][image8]

After applying threshold to eliminate false detections:

![alt text][image9]

![alt text][image10]

Images above show that very few number of overlapping windows are detected, and applying threshold to eliminate false detections would also eliminate the correct detections.

**Thus, finally I implement Hog Sub-Sampling Combined with Various Sliding Window Searches with below parameters:**
```python
ystart = [400, 416, 400, 432, 400, 432, 400, 464]
ystop  = [464, 480, 496, 528, 528, 560, 596, 660]
scale  = [1.0, 1.0, 1.5, 1.5, 2.0, 2.0, 3.5, 3.5]
```
Applying different scales for different window sizes.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Finally, I searched on four scales (1x, 1.5x, 2.0x, and 3.5x) using YUV 3-channel HOG features  in the feature vector, which provided a nice result. Below are the example images processed through the pipeline.

In **2) b)** of **Step 5: Apply Functions on Test Images for Vehicle Detection** of IPython notebook `P5_VehicleDetection.ipynb`, we see that several windows of different scales belong to correct detections:

![alt text][image16]

In **3)** of **Step 5: Apply Functions on Test Images for Vehicle Detection** of IPython notebook `P5_VehicleDetection.ipynb`, we next convert it to heatmap image. The `add_heat` function increments the pixel value (referred to as "heat") of an all-black image the size of the original image at the location of each detected window. Areas encompassed by more overlapping windows are assigned higher levels of heat. The following image is the resulting heatmap from the detections in the image above:

![alt text][image17]

In **4)** of **Step 5: Apply Functions on Test Images for Vehicle Detection** of IPython notebook `P5_VehicleDetection.ipynb`, we then apply threshold to eliminate false detections. A threshold is applied to the heatmap (in this example, with a value of 1), setting all pixels that don't exceed the threshold to zero. The result is below:

![alt text][image18]

In **5)** of **Step 5: Apply Functions on Test Images for Vehicle Detection** of IPython notebook `P5_VehicleDetection.ipynb`, the scipy.ndimage.measurements.label() function collects spatially contiguous areas of the heatmap and assigns each a label. And the final detection area is set to the extremities of each identified label:

![alt text][image19]

The steps / approach for classifier optimization is described in the Question 3 of Histogram of Oriented Gradients (HOG).

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Finally, I build a pipeline by combining all functions described above in `process_frame_for_video` function and then tested my pipeline on the `project_video.mp4`.

Here's a [link to my video result](https://github.com/viralzaveri12/CarND-P5-VehicleDetection/tree/master/output_videos "Project Videos with Detected Vehicles")

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

In **Section a)** of **Step 6: Build Pipeline for Vehicle Detection on Test Videos** of IPython notebook `P5_VehicleDetection.ipynb`, running the function `process_video` on `test_video.mp4` produces the output video  `project_test_video_output.mp4` which shows that vehicle detections (windows) are highly unsteady.

To optimize the pipeline for video, I recorded the positions of positive detections in each frame of the video in **Section b) Define a Class to Store Data from Vehicle Detections**. This stores the previous 15 frames using the `prev_windows` parameter from a class called `vehicle_detect`. In **Section c)**, rather than performing the heatmap/threshold/label steps for the current frame's detections, the detections for the past 15 frames are combined and added to the heatmap and the threshold for the heatmap is set to 1 + len(det.prev_windows)//2 (one more than half the number of window sets contained in the history) - this value was found to perform best empirically (rather than using a single scalar, or the full number of window sets in the history).

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The pipeline could fail in cases where vehicles (or the HOG features thereof) do not resemble those in the training dataset. Features from varying lighting and environmental conditions (e.g. a white car against a white background) would be an important factor in obtaining the decision boundary while training the classifier.

Using smaller window scales for distant vehicles tended to produce more false detections, and yet, still did not often correctly detect the smaller, distant cars.

I believe that the best approach, given plenty of time to pursue it, would be to combine a very high accuracy classifier with high overlap in the search windows. The execution cost could be offset with more intelligent tracking strategies, such as:

* determine vehicle location and speed to predict its location in subsequent frames
* begin with expected vehicle locations and nearest (largest scale) search areas, and preclude overlap and redundant detections from smaller scale search areas to speed up execution
* use a convolutional neural network, to preclude the sliding window search altogether

