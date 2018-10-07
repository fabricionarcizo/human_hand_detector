#!/usr/bin/python

#<!--------------------------------------------------------------------------->
#<!--             Segment Skin in Images Depicting Human Hands              -->
#<!-- File       : 03_human_hand_detector.py                                -->
#<!-- Description: This Python script uses supervised machine learning      -->
#<!--              methods to detect the skin color in human hand images    -->
#<!-- Author     : Fabricio Batista Narcizo                                 -->
#<!--            : Kastruplundgade 52, ST. MF., DK-2770, Kastrup, Danmark   -->
#<!-- Contact    : fabricionarcizo[at]gmail[dot]com                         -->
#<!-- Information: No additional information                                -->
#<!-- Date       : 06/10/2018                                               -->
#<!-- Change     : 06/10/2018 - Creation of this script                     -->
#<!-- Review     : 07/10/2018 - Finalized                                   -->
#<!--------------------------------------------------------------------------->

__version__ = "$Revision: 2018100701 $"

###############################################################################
import argparse
import cv2
import os
import numpy as np

from glob import glob

from sklearn.externals import joblib

###############################################################################

#-----------------------------------------------------------------------------#
#                          Image Analysis Functions                           #
#-----------------------------------------------------------------------------#
def createMask(image, model):
    """Create a binary mask using a supervised classifier."""

    # Get the image dimensions.
    h, w, d = image.shape

    # Convert the RGB or HSV values in a matrix of features.
    features = image.reshape(h * w, d)

    # Binary classification (0: Non-Skin, 1: Skin).
    mask = logreg.predict(features)
    mask = mask.reshape(h, w).astype(np.uint8)
    mask[mask == 1] = 255

    # Return a binary mask.
    return mask


#-----------------------------------------------------------------------------#
#                                Main Function                                #
#-----------------------------------------------------------------------------#
if __name__== "__main__":
    """Main function of the Python script."""

    # Define the script arguments.
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required=False,
                    help="Path to the dataset folder")
    args = vars(ap.parse_args())

    # Define the folder path and check if it exists.
    path = "models" if args["path"] is None else args["path"]
    if not os.path.exists(path):
        exit(0)

    # Load the machine learning models.
    logreg = joblib.load(path + "/LogisticRegression.joblib")
    tree = joblib.load(path + "/DecisionTree.joblib")
    svm = joblib.load(path + "/SVM.joblib")
    mlp = joblib.load(path + "/MLP.joblib")

    # Search for all JPEG files.
    filepaths = glob("datasets/*.jpg")
    filepaths.sort()

    # Load each image from the dataset.
    for filepath in filepaths:

        # Load a single image.
        image = cv2.imread(filepath)
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Logistic Regression.
        mask = createMask(hsv_image, logreg)
        result_01 = cv2.resize(mask, (0, 0), fx=0.10, fy=0.10)
        cv2.putText(result_01, "Logistic Regression", (10, 20),
                    cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))

        # Decision Tree.
        mask = createMask(hsv_image, tree)
        result_02 = cv2.resize(mask, (0, 0), fx=0.10, fy=0.10)
        cv2.putText(result_02, "Decision Tree", (10, 20),
                    cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))

        # SVM.
        mask = createMask(hsv_image, svm)
        result_03 = cv2.resize(mask, (0, 0), fx=0.10, fy=0.10)
        cv2.putText(result_03, "SVM", (10, 20),
                    cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))

        # Neural Network (MLP).
        mask = createMask(hsv_image, mlp)
        result_04 = cv2.resize(mask, (0, 0), fx=0.10, fy=0.10)
        cv2.putText(result_04, "Neural Network (MLP)", (10, 20),
                    cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))

        # Concatenate four different results.
        final = np.vstack((np.hstack((result_01, result_02)),
                           np.hstack((result_03, result_04))))

        # Show the final results.
        cv2.imshow("image", final)
        cv2.waitKey(0)

    # Open a video capture device using OpenCV.
    camera = cv2.VideoCapture(0)

    # Grab a sequence of image from the video streaming.
    while True:
        ret, image = camera.read()
        if not ret:
            break

        # Pre-processing.
        image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        #hsv_image = cv2.GaussianBlur(hsv_image, (5, 5), 0)

        # Get the image dimensions.
        h, w, d = hsv_image.shape

        # Logistic Regression.
        mask = createMask(hsv_image, logreg)
        result_01 = cv2.bitwise_and(image, image, mask=mask)
        cv2.putText(result_01, "Logistic Regression", (10, 20),
                    cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))

        # Decision Tree.
        mask = createMask(hsv_image, tree)
        result_02 = cv2.bitwise_and(image, image, mask=mask)
        cv2.putText(result_02, "Decision Tree", (10, 20),
                    cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))

        # SVM.
        mask = createMask(hsv_image, svm)
        result_03 = cv2.bitwise_and(image, image, mask=mask)
        cv2.putText(result_03, "SVM", (10, 20),
                    cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))

        # Neural Network (MLP).
        mask = createMask(hsv_image, mlp)
        result_04 = cv2.bitwise_and(image, image, mask=mask)
        cv2.putText(result_04, "Neural Network (MLP)", (10, 20),
                    cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))

        # Concatenate four different results.
        final = np.vstack((np.hstack((result_01, result_02)),
                           np.hstack((result_03, result_04))))

        # Show the final results.
        cv2.imshow("image", final)
        if (cv2.waitKey(1) == ord("q")):
            break
        
    # Release everything if job is finished.
    camera.release()
    cv2.destroyAllWindows()