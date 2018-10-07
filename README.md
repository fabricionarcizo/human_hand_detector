This project aims to segment the skin in images depicting human hands. The solution is based on the analysis of RGB and HSV color spaces in 4 images (i.e., 1.jpg, 2.jpg, 3.jpg, and 4.jpg) available in the folder `dataset`.

## 00_manual_segmentation.py
I have decided to create a Python script to define a binary threshold manually based on 2 (two) color spaces. This script converts each pixel of an input image into a row with the RGB (3 numbers), HSV (3 numbers) and the label (0: non-skin or 1: skin). You can use one of the images available in the folder `dataset` to generate the dataset as a text file.

This Python scripts contains an user interface (see Figure bellow) to help you segment the human skin. It has (6) six trackbars, namely: (1) `RGB <=> HSV`, choose the RGB or HSV color spaces; (2) `R <=> H`, select the Red or Hue values; (3) `G <=> S`, select the Green or Saturation values; (4) `B <=> V`, select the Blue or Value values; (5) `Lower Range` and (6) `Upper Range`, to define the lower and upper boundary parameters used by the function `cv2.inRange()`.

![Manual Segmentation](/images/00_manual_segmentation.png)

Execute the following command to run this script:
```
python 00_manual_segmentation.py --image "datasets/1.jpg"
```

Then, move the trackbars until you define a good mask to segment the user's hand (see the OpenCV windows `binary` and `mask`). When you are satisfied with the final result, press the button `s` of your keyboard to save the text file with the RGB and HSV values. The script saves the text file in the same folder of the input image. The dataset contains a sequence of number which represents: R,G,B,H,S,V,label.

__Info__: In my experiments, the best option was to use the HSV color space because I could segment the user's hand and remove the entire background.

## 01_plot_dataset.py
I have evaluated if it is feasible to use RGB and HSV color spaces to develop a skin color detector. This Python scripts plots 3 (three) graphics based on: (1) RGB color space [3D]; (2) HSV color space [3D]; and (3) RGB+HSV [6D]. Each plot shows the `non-skin` (red) and `skin` (blue) classes, and the decision boundary (plane) based on Logistic Regression.

As each dataset has information about 7.990.272 pixels, I have decided to use Pandas to speed up the file reading process. I also have decided to use a Gaussian sample of 1.000 pixels to generate and render graphics using Matplotlib.

__Hint__: You can change the sample size on line 58 [`sample = dataset.sample(1000, random_state=1)`].

I have used dimensionality reduction to plot the RGB+HSV data into 3D. The figure below shows an example of one plot generated with this Python script. It presents that the dataset contains two different clusters to represent non-skin and skin pixels. The grey plane represents the decision boundary used to classify the pixels into these two classes. According to my initial evaluation, it is possible to use any of the 3 (three) approaches mentioned above as `features` to segment the skin color in images depicting human hands.
![Plot Dataset](/images/01_plot_dataset.png)

Execute the following command to run this script:
```
python 01_plot_dataset.py --dataset "datasets/1.txt"
```

## 02_training_classifiers.py
I have decided to use 5 (five) different Supervised Machine Learning (ML) methods for binary classification, namely: (1) Logistic Regression; (2) K-Nearest Neighbors (KNN); (3) Decision Tree; (4) Support Vector Machine (SVM); and (5) Neural Network (NN) based on Multi Layer Perception (MLP). I have used `scikit-learn` library to create, train and evaluate the skin color classifiers.

This Python scripts uses one of the datasets generated with the Python script `00_manual_segmentation.py`. Some ML methods are fast to process of 7.990.272 pixels in a short time. However, others methods (i.e., KNN, SVM) are very slow to process the entire dataset. Therefore, you can change the sample size on line 208. 

I have evaluated each ML method for RGB, HSV and RGB+HSV features. I calculated the `accuracy` and `confusion matrix` to evaluate the ML method using 70% of training data and 30% of test data. The sample in this analysis had 100.000 pixels from the dataset `./datasets/1.txt`. All methods have achieved and accuracy bigger than 99%. The best one was `Decision Tree` using HSV and RGB+HSV colorspaces, in which it has achieved 100% of accuracy. See bellow the results of my evaluation:

### LOGISTIC REGRESSION (RGB)
Accuracy: 0.9991666666666666

Confusion Matrix:

| | Class 1<br/>Predicted | Class 2<br/>Predicted |
| :-: | :-: | :-: |
| __Class 1<br/>Actual__ | 21138 |   13 |
| __Class 2<br/>Actual__ |    12 | 8837 |

### LOGISTIC REGRESSION (HSV)
Accuracy: 0.9999666666666667

Confusion Matrix:

| | Class 1<br/>Predicted | Class 2<br/>Predicted |
| :-: | :-: | :-: |
| __Class 1<br/>Actual__ | 21150 |    1 |
| __Class 2<br/>Actual__ |     0 | 8849 |

### LOGISTIC REGRESSION (RGB+HSV)
Accuracy: 0.9999

Confusion Matrix:

| | Class 1<br/>Predicted | Class 2<br/>Predicted |
| :-: | :-: | :-: |
| __Class 1<br/>Actual__ | 21148 |    3 |
| __Class 2<br/>Actual__ |     0 | 8849 |

### K-NEAREST NEIGHBORS (RGB)
Accuracy: 0.9990666666666667

Confusion Matrix:

| | Class 1<br/>Predicted | Class 2<br/>Predicted |
| :-: | :-: | :-: |
| __Class 1<br/>Actual__ | 21139 |   12 |
| __Class 2<br/>Actual__ |    16 | 8833 |

### K-NEAREST NEIGHBORS (HSV)
Accuracy: 0.9997333333333334

Confusion Matrix:

| | Class 1<br/>Predicted | Class 2<br/>Predicted |
| :-: | :-: | :-: |
| __Class 1<br/>Actual__ | 21148 |    3 |
| __Class 2<br/>Actual__ |     5 | 8844 |

### K-NEAREST NEIGHBORS (RGB+HSV)
Accuracy: 0.9996

Confusion Matrix:

| | Class 1<br/>Predicted | Class 2<br/>Predicted |
| :-: | :-: | :-: |
| __Class 1<br/>Actual__ | 21147 |    4 |
| __Class 2<br/>Actual__ |     8 | 8841 |

### DECISION TREE (RGB)
Accuracy: 0.9992

Confusion Matrix:

| | Class 1<br/>Predicted | Class 2<br/>Predicted |
| :-: | :-: | :-: |
| __Class 1<br/>Actual__ | 21145 |    6 |
| __Class 2<br/>Actual__ |    18 | 8831 |

### DECISION TREE (HSV)
Accuracy: 1.0

Confusion Matrix:

| | Class 1<br/>Predicted | Class 2<br/>Predicted |
| :-: | :-: | :-: |
| __Class 1<br/>Actual__ | 21151 |    3 |
| __Class 2<br/>Actual__ |     0 | 8849 |

### DECISION TREE (RGB+HSV)
Accuracy: 1.0

Confusion Matrix:

| | Class 1<br/>Predicted | Class 2<br/>Predicted |
| :-: | :-: | :-: |
| __Class 1<br/>Actual__ | 21151 |    0 |
| __Class 2<br/>Actual__ |     0 | 8849 |

### SUPPORT VECTOR MACHINE (RGB)
Accuracy: 0.9973666666666666

Confusion Matrix:

| | Class 1<br/>Predicted | Class 2<br/>Predicted |
| :-: | :-: | :-: |
| __Class 1<br/>Actual__ | 21075 |   76 |
| __Class 2<br/>Actual__ |     3 | 8846 |

### SUPPORT VECTOR MACHINE (HSV)
Accuracy: 0.9947

Confusion Matrix:

| | Class 1<br/>Predicted | Class 2<br/>Predicted |
| :-: | :-: | :-: |
| __Class 1<br/>Actual__ | 20992 |  159 |
| __Class 2<br/>Actual__ |     0 | 8849 |

### SUPPORT VECTOR MACHINE (RGB+HSV)
Accuracy: 0.9939

Confusion Matrix:

| | Class 1<br/>Predicted | Class 2<br/>Predicted |
| :-: | :-: | :-: |
| __Class 1<br/>Actual__ | 20969 |  182 |
| __Class 2<br/>Actual__ |     1 | 8848 |

### Multi Layer Perception (RGB)
Accuracy: 0.9993

Confusion Matrix:

| | Class 1<br/>Predicted | Class 2<br/>Predicted |
| :-: | :-: | :-: |
| __Class 1<br/>Actual__ | 21138 |   13 |
| __Class 2<br/>Actual__ |     8 | 8841 |

### Multi Layer Perception (HSV)
Accuracy: 0.9987333333333334

Confusion Matrix:

| | Class 1<br/>Predicted | Class 2<br/>Predicted |
| :-: | :-: | :-: |
| __Class 1<br/>Actual__ | 21120 |   31 |
| __Class 2<br/>Actual__ |     7 | 8842 |

### Multi Layer Perception (RGB+HSV)
Accuracy: 0.9992666666666666

Confusion Matrix:

| | Class 1<br/>Predicted | Class 2<br/>Predicted |
| :-: | :-: | :-: |
| __Class 1<br/>Actual__ | 21133 |   18 |
| __Class 2<br/>Actual__ |     4 | 8845 |

Execute the following command to run this script:
```
python 02_training_classifiers.py --dataset "datasets/1.txt"
```

## 03_human_hand_detector.py
Finally, this Python script detects and segments the user's hand based on HSV color space. It uses the models trained with the script `02_training_classifiers.py`. Execute the following command to run this script:
```
python 03_human_hand_detector.py --path "models"
```

First, it uses the 4 images from the dataset (i.e., 1.jpg, 2.jpg, 3.jpg, and 4.jpg). Figure bellow shows the results of `Logistic Regression`, `Decision Tree`, `SVM` and `Multi Layer Perception` using the image `3.jpg`. As you can see, the detector is able to segment the user's hand and remove the entire background from the image.
![Human Hand Detector](/images/03_human_hand_detector.png)

__Hint__: Press any button from your keyboard to go to the next image.

After showing all images from the dataset, the Python scripts opens the video capture device and uses the detector to segment the hands of images from your main webcam. You have to press the button `q` to quit the program.

## Future Improvements
To improve the skin detector, it is necessary to perform some additional task, e.g.:
1. Join the features of all images from the dataset in a single dataset
2. Add more images of human hand with different background and illumination conditions
3. Use pre-processing methods (e.g. spatial filtering, mathematical morphology) to remove noise from the input images
4. Include infrared images to handle
5. Adapt the detector to detect different skin colors