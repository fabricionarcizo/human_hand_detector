#!/usr/bin/python

#<!--------------------------------------------------------------------------->
#<!--                          LEO Innovation Lab                           -->
#<!--                    (https://leoinnovationlab.com)                     -->
#<!-- File       : 00_manual_segmentation.py                                -->
#<!-- Description: This Python script helps to select manually a threshold  -->
#<!--              based on the RGB and HSV color spaces                    -->
#<!-- Author     : Fabricio Batista Narcizo                                 -->
#<!--            : Kastruplundgade 52, ST. MF., DK-2770, Kastrup, Danmark   -->
#<!-- Contact    : fabricionarcizo[at]gmail[dot]com                         -->
#<!-- Information: No additional information                                -->
#<!-- Date       : 05/10/2018                                               -->
#<!-- Change     : 05/10/2018 - Creation of this script                     -->
#<!-- Review     : 06/10/2018 - Finalized                                   -->
#<!--------------------------------------------------------------------------->

__version__ = "$Revision: 2018100601 $"

###############################################################################
import argparse
import cv2
import os
import numpy as np

###############################################################################

#-----------------------------------------------------------------------------#
#                         OpenCV Trackbars Functions                          #
#-----------------------------------------------------------------------------#
def onChangeColorSpace(value):
    """Change the current color space."""
    global color_space
    color_space = value
    update()

def onChangeRH(value):
    """Change the values of the Red (R) or Hue (H) channels."""
    global rh_value
    rh_value = value
    update()

def onChangeGS(value):
    """Change the values of the Green (G) or Saturation (S) channels."""
    global gs_value
    gs_value = value
    update()

def onChangeBV(value):
    """Change the values of the Blue (B) or Value (V) channels."""
    global bv_value
    bv_value = value
    update()

def onChangeLowRange(value):
    """Change the lower boundary scalar."""
    global low_value
    low_value = value
    update()

def onChangeHighRange(value):
    """Change the upper boundary scalar."""
    global high_value
    high_value = value
    update()


#-----------------------------------------------------------------------------#
#                               Math Functions                                #
#-----------------------------------------------------------------------------#
def calculateLowerRange():
    """Calculate the lower range used by the function cv2.inRange()."""
    global lower_rh
    global lower_gs
    global lower_bv

    lower_rh = rh_value - low_value
    lower_gs = gs_value - low_value
    lower_bv = bv_value - low_value

    if lower_rh < 0:
        lower_rh = 0
    if lower_gs < 0:
        lower_gs = 0
    if lower_bv < 0:
        lower_bv = 0

def calculateUpperRange():
    """Calculate the upper range used by the function cv2.inRange()."""
    global upper_rh
    global upper_gs
    global upper_bv

    upper_rh = rh_value + high_value
    upper_gs = gs_value + high_value
    upper_bv = bv_value + high_value

    if upper_rh > 255:
        upper_rh = 255
    if upper_gs > 255:
        upper_gs = 255
    if upper_bv > 255:
        upper_bv = 255


#-----------------------------------------------------------------------------#
#                          Image Analysis Functions                           #
#-----------------------------------------------------------------------------#
def update():
    """Process the input image and update all OpenCV windows."""
    global thres

    # Calculate the lower and upper limits used by the function cv2.inRange().
    calculateLowerRange()
    calculateUpperRange()    

    # Select RGB or HSV images.
    color = hsv_image.copy() if color_space else rgb_image.copy()

    # Create a binary image based on a range of colors.
    thres = cv2.inRange(color, (lower_rh, lower_gs, lower_bv),
                               (upper_rh, upper_gs, upper_bv))

    # Show the binary image.
    resized = cv2.resize(thres, (0, 0), fx=0.15, fy=0.15)
    cv2.imshow("binary", resized)

    # Shows only the pixels of the human hand.
    mask = cv2.bitwise_and(image, image, mask=thres)
    resized = cv2.resize(mask, (0, 0), fx=0.15, fy=0.15)
    cv2.imshow("mask", resized)

def exportDataset(filepath):
    """Create and export the dataset based on features extracted from the input
       image."""

    # Define the output filename.
    filename = filepath.split(".")[0] + ".txt"

    # Get the image dimensions.
    h, w, d = rgb_image.shape

    # Convert the RGB and HSV values in a matrix of features.
    features = np.hstack((rgb_image.reshape(h * w, d),
                          hsv_image.reshape(h * w, d)))

    # Create the labels of pixels, i.e. 0: non-skin, and 1: skin.
    labels = thres.reshape(h * w, 1)
    labels[labels == 255] = 1

    # Skin segmentation dataset.
    dataset = np.hstack((features, labels))

    # Save the dataset.
    print("Saving the dataset...")
    np.savetxt(filename, dataset, fmt="%i", delimiter=",")
    print("Done!")

#-----------------------------------------------------------------------------#
#                                Main Function                                #
#-----------------------------------------------------------------------------#
if __name__== "__main__":
    """Main function of the Python script."""

    # Define the script arguments.
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=False,
                    help="Path to the input image")
    args = vars(ap.parse_args())

    # Define the filepath and check if the file exists.
    filepath = "dataset/1.jpg" if args["image"] is None else args["image"]
    if not os.path.exists(filepath):
        exit(0)

    # Global variables and their initial values.
    color_space = 0

    rh_value = 0
    gs_value = 0
    bv_value = 0

    low_value = 0
    high_value = 0

    lower_rh = 0
    lower_gs = 0
    lower_bv = 0
    upper_rh = 0
    upper_gs = 0
    upper_bv = 0
    
    # Define the main OpenCV window with 6 (six) trackbars.
    cv2.namedWindow("image")
    cv2.createTrackbar("RGB <=> HSV", "image", 0, 1, onChangeColorSpace)
    cv2.createTrackbar("R <=> H", "image", 0, 255, onChangeRH)
    cv2.createTrackbar("G <=> S", "image", 0, 255, onChangeGS)
    cv2.createTrackbar("B <=> V", "image", 0, 255, onChangeBV)
    cv2.createTrackbar("Low Range", "image", 0, 255, onChangeLowRange)
    cv2.createTrackbar("High Range", "image", 0, 255, onChangeHighRange)

    # Open the input image.
    image = cv2.imread(filepath)

    # Create images with different color spaces.
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the text to show to the user.
    text = "1: Select the color space (RGB or HSV)\n" + \
           "2: Change the trackbars to define a good mask\n" + \
           "3: Press (s) to create and save the dataset"

    # Add the text in the final image.
    resized = cv2.resize(image, (0, 0), fx=0.175, fy=0.175)
    y0, dy = 20, 15
    for i, line in enumerate(text.split("\n")):
        y = y0 + i * dy
        cv2.putText(resized, line, (10, y), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))

    # Show the input image as a resized image.
    cv2.imshow("image", resized)
    key = cv2.waitKey(0)

    # Save the image as a text file.
    if key == ord("s"):
        exportDataset(filepath)