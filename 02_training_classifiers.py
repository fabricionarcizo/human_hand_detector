#!/usr/bin/python

#<!--------------------------------------------------------------------------->
#<!--             Segment Skin in Images Depicting Human Hands              -->
#<!-- File       : 02_training_classifiers.py                               -->
#<!-- Description: This Python script trains a set of classifiers for skin  -->
#<!--              color detectors based on the RGB and HSV color spaces    -->
#<!-- Author     : Fabricio Batista Narcizo                                 -->
#<!--            : Kastruplundgade 52, ST. MF., DK-2770, Kastrup, Danmark   -->
#<!-- Contact    : fabricionarcizo[at]gmail[dot]com                         -->
#<!-- Information: No additional information                                -->
#<!-- Date       : 06/10/2018                                               -->
#<!-- Change     : 06/10/2018 - Creation of this script                     -->
#<!-- Review     : 06/10/2018 - Finalized                                   -->
#<!--------------------------------------------------------------------------->

__version__ = "$Revision: 2018100601 $"

###############################################################################
import argparse
import os
import numpy as np
import pandas as pd

from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

###############################################################################

#-----------------------------------------------------------------------------#
#                              Models Functions                               #
#-----------------------------------------------------------------------------#

def logisticRegression(dataset, labels, name="Default", save=False):
    """Train and evaluate a classifier using Logistic Regression."""

    # Logistic Regression.
    logreg = LogisticRegression(solver="lbfgs", max_iter=6000)

    # Split the dataset into a training and test subset.
    x_train, x_test, y_train, y_test = train_test_split(dataset, labels,
                                                        test_size=0.3,
                                                        random_state=1)    

    # Fit the model using the training subset.
    logreg.fit(x_train, y_train)

    # Evaluate the test subset.
    predictions = logreg.predict(x_test)
    score = logreg.score(x_test, y_test)    

    # Print the report.
    print("##### LOGISTIC REGRESSION (%s) #####" % name)
    print("Accuracy: " + str(score))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))
    print("")

    # Model persistence.
    if save:
        joblib.dump(logreg, "models/LogisticRegression.joblib")

def knn(dataset, labels, name="Default", save=False):
    """Train and evaluate a classifier using K-Nearest Neighbors."""

    # Classifier implementing the k-nearest neighbors vote.
    knn = KNeighborsClassifier()

    # Split the dataset into a training and test dataset.
    x_train, x_test, y_train, y_test = train_test_split(dataset, labels,
                                                        test_size=0.3,
                                                        random_state=1)    

    # Fit the model using the training subset.
    knn.fit(x_train, y_train)

    # Evaluate the test subset.
    predictions = knn.predict(x_test)
    score = knn.score(x_test, y_test)    

    # Print the report.
    print("##### K-NEAREST NEIGHBORS (%s) #####" % name)
    print("Accuracy: " + str(score))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))
    print("")

    # Model persistence.
    if save:
        joblib.dump(knn, "models/KNeighbors.joblib")

def decisionTree(dataset, labels, name="Default", save=False):
    """Train and evaluate a classifier using Decision Tree."""

    # A decision tree classifier.
    tree = DecisionTreeClassifier()

    # Split the dataset into a training and test dataset.
    x_train, x_test, y_train, y_test = train_test_split(dataset, labels,
                                                        test_size=0.3,
                                                        random_state=1)    

    # Fit the model using the training subset.
    tree.fit(x_train, y_train)

    # Evaluate the test subset.
    predictions = tree.predict(x_test)
    score = tree.score(x_test, y_test)    

    # Print the report.
    print("##### DECISION TREE (%s) #####" % name)
    print("Accuracy: " + str(score))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))
    print("")

    # Model persistence.
    if save:
        joblib.dump(tree, "models/DecisionTree.joblib")

def supportVectorMachine(dataset, labels, name="Default", save=False):
    """Train and evaluate a classifier using Support Vector Machine."""

    # A decision tree classifier.
    svm = SVC(gamma="auto")

    # Split the dataset into a training and test dataset.
    x_train, x_test, y_train, y_test = train_test_split(dataset, labels,
                                                        test_size=0.3,
                                                        random_state=1)    

    # Fit the model using the training subset.
    svm.fit(x_train, y_train)

    # Evaluate the test subset.
    predictions = svm.predict(x_test)
    score = svm.score(x_test, y_test)    

    # Print the report.
    print("##### SUPPORT VECTOR MACHINE (%s) #####" % name)
    print("Accuracy: " + str(score))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))
    print("")

    # Model persistence.
    if save:
        joblib.dump(svm, "models/SVM.joblib")

def multiLayerPerception(dataset, labels, name="Default", save=False):
    """Train and evaluate a classifier using Multi Layer Perception."""

    # A decision tree classifier.
    mlp = MLPClassifier(hidden_layer_sizes=(13, 13, 13),max_iter=500)

    # Split the dataset into a training and test dataset.
    x_train, x_test, y_train, y_test = train_test_split(dataset, labels,
                                                        test_size=0.3,
                                                        random_state=1)    

    # Fit the model using the training subset.
    mlp.fit(x_train, y_train)

    # Evaluate the test subset.
    predictions = mlp.predict(x_test)
    score = mlp.score(x_test, y_test)    

    # Print the report.
    print("##### Multi Layer Perception (%s) #####" % name)
    print("Accuracy: " + str(score))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))
    print("")

    # Model persistence.
    if save:
        joblib.dump(mlp, "models/MLP.joblib")


#-----------------------------------------------------------------------------#
#                                Main Function                                #
#-----------------------------------------------------------------------------#
if __name__== "__main__":
    """Main function of the Python script."""

    # Define the script arguments.
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=False,
                    help="Path to the dataset")
    args = vars(ap.parse_args())

    # Define the filepath and check if the file exists.
    filepath = "datasets/1.txt" if args["dataset"] is None else args["dataset"]
    if not os.path.exists(filepath):
        exit(0)

    # Using pandas to speed up the reading process of the text file.
    names = ["R", "G", "B", "H", "S", "V", "label"]
    dataset = pd.read_csv(filepath, names=names)

    # Select a sample of 100000 from the dataset.
    dataset = dataset.sample(100000, random_state=1)

    # Create a subset of the RGB and HSV color spaces.
    rgb_data = dataset.loc[:, "R":"B"].values
    hsv_data = dataset.loc[:, "H":"V"].values
    color_data = dataset.loc[:, "R":"V"].values

    # Labels: 0 "Non-Skin" and 1 "Skin".
    labels = dataset["label"].values

    # Logistic Regression for RGB color space.
    logisticRegression(rgb_data, labels, "RGB")

    # Logistic Regression for HSV color space.
    logisticRegression(hsv_data, labels, "HSV")

    # Logistic Regression for RGB+HSV color space.
    logisticRegression(color_data, labels, "RGB+HSV")

    # K-Nearest Neighbors for RGB color space.
    knn(rgb_data, labels, "RGB")

    # K-Nearest Neighbors for HSV color space.
    knn(hsv_data, labels, "HSV")

    # K-Nearest Neighbors for RGB+HSV color space.
    knn(color_data, labels, "RGB+HSV")

    # Decision Tree for RGB color space.
    decisionTree(rgb_data, labels, "RGB")

    # Decision Tree for HSV color space.
    decisionTree(hsv_data, labels, "HSV")

    # Decision Tree for RGB+HSV color space.
    decisionTree(color_data, labels, "RGB+HSV")

    # Support Vector Machine for RGB color space.
    supportVectorMachine(rgb_data, labels, "RGB")

    # Support Vector Machine for HSV color space.
    supportVectorMachine(hsv_data, labels, "HSV")

    # Support Vector Machine for RGB+HSV color space.
    supportVectorMachine(color_data, labels, "RGB+HSV")

    # Multi Layer Perception for RGB color space.
    multiLayerPerception(rgb_data, labels, "RGB")

    # Multi Layer Perception for HSV color space.
    multiLayerPerception(hsv_data, labels, "HSV")

    # Multi Layer Perception for RGB+HSV color space.
    multiLayerPerception(color_data, labels, "RGB+HSV")
