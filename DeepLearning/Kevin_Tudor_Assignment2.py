# -*- coding: utf-8 -*-
"""
Introduction to Deep Learning – CAP 4613
Assignment 2
Due: Sunday, February 6 before 11:59 pm

Name: Kevin Tudor z-number:Z23468207

Spyder (Python 3.8) IDE

link to final version in Colab:
https://colab.research.google.com/drive/1tC6UXuBWwDlW4Nn4wau1J4QYSX1O02qI?usp=sharing
"""
import webbrowser
from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np

url = 'https://colab.research.google.com/drive/1tC6UXuBWwDlW4Nn4wau1J4QYSX1O02qI?usp=sharing'
webbrowser.open(url)  # Go to the google colab

"""
Problem 3) 
    
MNIST dataset - The MNIST dataset is divided into two sets - training and test. Each
set comprises a series of images (28 x 28-pixel images of handwritten digits) and their
respective labels (values from 0 - 9, representing which digit the image corresponds to).
"""

"""
Part b)
Write a function (with images of ten digits and labels as the input) that plots a figure
with 10 subplots for each 0-9 digits. Each subplot has the number of the handwritten
digit in the title. 
"""

def plot(x, y, t):
    
    for i in range(10):
        plt.suptitle(t)
        plt.subplot(2,5,i + 1)
        plt.imshow(x[i,:,:], cmap = 'gray')
        plt.title('Label: ' + str(i))
    
    plt.show()

def main():
    
    """
    Part a)
        
    Use mnist function in keras.datasets to split the MNIST dataset into the training and
    testing sets. Print the following: The number of images in each training and testing
    set, and the image width and height.
    """

    #x_train is training (images, width, height)
    #y_train is training (labels)
    #x_test is testing (images, width, height)
    #y_test is testing (labels)

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    print("\nPart a)\n")
    print("# of images in x_train:", x_train.shape[0], "\n# of images in y_train (labels):", y_train.shape[0])
    print("\nx_train image shape:", x_train.shape[1:], "\ny_train (labels) image shape:", y_train.shape[1:])
    print("\n# of images in x_test:", x_test.shape[0], "\n# of images in y_test (labels):", y_test.shape[0])
    print("\nx_test image shape:", x_test.shape[1:], "\ny_test (labels) image shape:", y_test.shape[1:])
    print("\n END Part a)\n")
    
    #shuffling training set
    num_train_img = x_train.shape[0]
    train_ind = np.arange(0, num_train_img)
    
    train_ind_s = np.random.permutation(train_ind)
    
    x_train = x_train[train_ind_s,:,:]
    y_train = y_train[train_ind_s]
    
    """
    Part d)
    
    In machine learning, we usually divide the training set into two sets of training and 
    validation sets to adjust a machine learning model parameters. In your code, 
    randomly select 20% of the training images and their corresponding labels and name 
    them as x_valid and y_valid, respectively. Name the remaining training images and 
    their labels as x_train and y_train, respectively. Print the number of images in each 
    training and validation set. Note: that there are no overlaps between the two sets. 
    """
    #Choose the percent to split by
    split_per = 20
    split_per = split_per/100

    #select 20% of train data as validation
    x_valid = x_train[0:int(split_per*num_train_img),:,:]
    y_valid = y_train[0:int(split_per*num_train_img)]
    
    #the rest of the train data 
    x_train = x_train[int(split_per*num_train_img):,:,:]
    y_train = y_train[int(split_per*num_train_img):]
    
    
    """
    Part c)
    
    Create a loop to call the plot function in (b) with images from each set to create three 
    figures. Note: the code has to select the images randomly. Include all the 10 digits in 
    each figure. Show the results of your code. 
    """ 
    images = np.zeros((10, 28, 28))
    labels = np.zeros((10))
    
    for i in range (10):
        tmp = x_train[y_train == i,:,:]
        images[i, :, :] = tmp[0,:,:]
        labels[i] = i
        title = "Training Data"
    
    plot(images, labels, title)
    
    for i in range (10):
        tmp = x_test[y_test == i,:,:]
        images[i, :, :] = tmp[0,:,:]
        labels[i] = i
        title = "Testing Data"
    
    plot(images, labels, title)
    
    for i in range (10):
        tmp = x_valid[y_valid == i,:,:]
        images[i, :, :] = tmp[0,:,:]
        labels[i] = i
        title = "Validation Data"
    
    plot(images, labels, title)

    print("\nPart d)\n")
    print("# of images in x_train:", x_train.shape[0], "\n# of images in y_train (labels):", y_train.shape[0])
    print("\nx_train image shape:", x_train.shape[1:], "\ny_train (labels) image shape:", y_train.shape[1:])
    print("\n# of images in x_test:", x_test.shape[0], "\n# of images in y_test (labels):", y_test.shape[0])
    print("\nx_test image shape:", x_test.shape[1:], "\ny_test (labels) image shape:", y_test.shape[1:])
    print("\n# of images in x_valid:", x_valid.shape[0], "\n# of images in y_valid (labels):", y_valid.shape[0])
    print("\nx_valid image shape:", x_valid.shape[1:], "\ny_valid (labels) image shape:", y_valid.shape[1:])
    print("\n END Part d)\n")

print("-----------------------------------------------------")
    
if (__name__=="__main__"):
    
    main()


print("-----------------------------------------------------")
