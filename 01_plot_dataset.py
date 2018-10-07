#!/usr/bin/python

#<!--------------------------------------------------------------------------->
#<!--             Segment Skin in Images Depicting Human Hands              -->
#<!-- File       : 01_plot_dataset.py                                       -->
#<!-- Description: This Python script plots the human hands dataset based   -->
#<!--              the RGB and HSV color spaces distribuitions              -->
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

from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE

import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D

###############################################################################

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

    # Select a sample of 1000 from the dataset.
    sample = dataset.sample(1000, random_state=1)

    # Create a subset of the RGB and HSV color spaces.
    rgb_data = sample.loc[:, "R":"B"].values
    hsv_data = sample.loc[:, "H":"V"].values
    color_data = sample.loc[:, "R":"V"].values

    # Labels: 0 "Non-Skin" and 1 "Skin".
    labels = sample["label"].values

    # Logistic Regression (used to plot the decision boundary).
    logreg = LogisticRegression(solver="lbfgs")

    # Decision boundary of RGB subset.
    logreg.fit(rgb_data, labels)
    intercept = logreg.intercept_[0]
    coeff = logreg.coef_[0]

    z = lambda x,y: (-intercept - coeff[0] * x - coeff[1] * y) / coeff[2]
    tmp = np.linspace(rgb_data.min(), rgb_data.max(), 50)
    x, y = np.meshgrid(tmp, tmp)

    # Custom legend.
    custom_markers = [Line2D([0], [0], color="w", marker="o", markerfacecolor="r"),
                      Line2D([0], [0], color="w", marker="^", markerfacecolor="b")]

    # Plot the RGB subset.
    fig = plt.figure()
    ax_rgb = fig.add_subplot(111, projection="3d")
    ax_rgb.plot_surface(x, y, z(x, y), alpha=0.2)

    for i, c, m in zip(range(2), ("r", "b"), ("o", "^")):
        xs = rgb_data[:, 0][labels == i]
        ys = rgb_data[:, 1][labels == i]
        zs = rgb_data[:, 2][labels == i]
        ax_rgb.scatter(xs, ys, zs, c=c, marker=m)

    ax_rgb.set_title("RGB Channels")
    ax_rgb.set_xlabel("R Channel")
    ax_rgb.set_ylabel("G Channel")
    ax_rgb.set_zlabel("B Channel")
    ax_rgb.legend(custom_markers, ["No Skin", "Skin"])

    # Decision boundary of HSV subset.
    logreg.fit(hsv_data, labels)
    intercept = logreg.intercept_[0]
    coeff = logreg.coef_[0]

    tmp = np.linspace(hsv_data.min(), hsv_data.max(), 50)
    x, y = np.meshgrid(tmp, tmp)

    # Plot the HSV subset.
    fig = plt.figure()
    ax_hsv = fig.add_subplot(111, projection="3d")
    ax_hsv.plot_surface(x, y, z(x, y), alpha=0.2)

    for i, c, m in zip(range(2), ("r", "b"), ("o", "^")):
        xs = hsv_data[:, 0][labels == i]
        ys = hsv_data[:, 1][labels == i]
        zs = hsv_data[:, 2][labels == i]
        ax_hsv.scatter(xs, ys, zs, c=c, marker=m)

    ax_hsv.set_title("HSV Channels")
    ax_hsv.set_xlabel("H Channel")
    ax_hsv.set_ylabel("S Channel")
    ax_hsv.set_zlabel("V Channel")
    ax_hsv.legend(custom_markers, ["No Skin", "Skin"])

    # Using dimensionality reduction to transform a 6D dataset to a 3D dataset.
    tsne = TSNE(n_components=3).fit_transform(color_data)

    # Decision boundary of RGB subset.
    logreg.fit(tsne, labels)
    intercept = logreg.intercept_[0]
    coeff = logreg.coef_[0]

    tmp = np.linspace(tsne.min(), tsne.max(), 50)
    x, y = np.meshgrid(tmp, tmp)

    # Plot the RGB+HSV dataset in 3D.
    fig = plt.figure()
    ax_tsne = fig.add_subplot(111, projection="3d")
    ax_tsne.plot_surface(x, y, z(x, y), alpha=0.2)

    for i, c, m in zip(range(2), ("r", "b"), ("o", "^")):
        xs = tsne[:, 0][labels == i]
        ys = tsne[:, 1][labels == i]
        zs = tsne[:, 2][labels == i]
        ax_tsne.scatter(xs, ys, zs, c=c, marker=m)

    ax_tsne.set_title("3D t-SNE")
    ax_tsne.set_xlabel("X")
    ax_tsne.set_ylabel("Y")
    ax_tsne.set_zlabel("Z")
    ax_tsne.legend(custom_markers, ["No Skin", "Skin"])

    # Plot the graphics.
    plt.show()
