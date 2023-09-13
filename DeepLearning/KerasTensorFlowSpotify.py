# -*- coding: utf-8 -*-
"""
Introduction to Deep Learning
Assignment 5
Problem 1-3
Due: Sunday, April 3 before 11:59 pm

Spyder (Python 3.8) IDE

link to final version in Colab:
https://colab.research.google.com/drive/1_4sazxvFT_66YceJnaSSYGKE8FXf38wv?usp=sharing
"""
import webbrowser

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
from tensorflow.python.keras.utils import np_utils
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np

#url = 'https://colab.research.google.com/drive/1_4sazxvFT_66YceJnaSSYGKE8FXf38wv?usp=sharing'
#webbrowser.open(url)  # Go to the google colab


#b) Plot the given data points with two different markers for each group. 
def plot_fun(features,labels,classes):
    
  plt.plot(features[labels[:]==classes[0],0], features[labels[:]==classes[0],1], 'rs', 
         features[labels[:]==classes[1],0], features[labels[:]==classes[1],1], 'g^',markersize=15)
  
  #plt.axis([-2,2,-2,2])
  plt.xlabel('x: feature 1')
  plt.ylabel('y: feature 2')
  plt.legend(['Class'+str(classes[0]), 'Class'+str(classes[1])])
  plt.show()

#g) Use the trained weights and plot the final classifier lines in the plot of part (b)
def plot_fun_thr(features,labels,thre_parms,classes):
  #ploting the data points
  plt.plot(features[labels[:]==classes[0],0], features[labels[:]==classes[0],1], 'rs',
           features[labels[:]==classes[1],0], features[labels[:]==classes[1],1], 'g^', 
           markersize=15)
  plt.axis([-2,2,-2,2])
  #ploting the seperating line
  x1 = np.linspace(-2,2,50)
  x2 = -(thre_parms[0]*x1+thre_parms[2])/thre_parms[1] 
                      #a X1 + b X2 + c=0 --> x2 = -(a X1 + c)/b
  plt.plot(x1, x2, '-r')
  plt.xlabel('x: feature 1')
  plt.ylabel('y: feature 2')
  plt.legend(['Class'+str(classes[0]), 'Class'+str(classes[1])])
  #plt.pause(0.5)
  #plt.show()

#h) Plot the training loss (i.e., the learning curve) for all the epochs.
def plot_curve(accuracy_train, loss_train):
  epochs=np.arange(loss_train.shape[0])
  plt.subplot(1,3,2)
  plt.plot(epochs,accuracy_train)
  plt.yticks(np.linspace(0, 1, 11))
  plt.xlabel('Epoch#')
  plt.ylabel('Accuracy')
  plt.title('Training Accuracy')

  plt.subplot(1,3,3)
  plt.plot(epochs,loss_train)
  plt.xlabel('Epoch#')
  plt.ylabel('Binary crossentropy loss')
  plt.title('Training loss')

  #plt.show()
  
"""
Problem 1) [Python] Application of Keras to build, compile, and train a neural network to
perform XOR operation:
""" 
def p_1():
    
    #plot_fun(features,labels,classes)
    
    """
    c) Based on the plot from part (b), what is the minimum number of layers and nodes that 
    is required to classify the training data points correctly? Explain.
    
    The Neural Network will need a minimum of 3 layers with a total of 6 nodes
    layers:    input layer       hidden layer 1       Output layer
    nodes:   3 nodes/inputs         2 nodes             1 node
    """
    
    #d) Build the network that you proposed in part c using the Keras library
    #Define (model, sequential, multi-GPU)
    model_a=Sequential()
    model_a.add(Dense(input_dim=2, units=4, activation='relu'))
    model_a.add(Dense(units=1, activation='sigmoid'))
    model_a.summary()
    
    """
    e)
    Compile the network. Make sure to select a correct loss function for this classification
    problem. Use stochastic gradient descent learning (SGD, learning rate of 0.1).
    Explain your selection of the loss function
    
    I selected binary_crossentropy for the loss function because the output will be binary(T/F)
    """
    #e) Compile (optimizer, loss, metrics)
    opt = tf.keras.optimizers.SGD(learning_rate=0.1)
    model_a.compile(loss='binary_crossentropy',
                    optimizer=opt,
                    metrics=['accuracy'])
    
    
    #Fit (Batch Size, epochs, Validation Split)
    #a) Create np array of shape 4x2 for the inputs and another 4x1 array for labels of XOR.
    features=np.array([[0,0], [0,1], [1,0], [1,1]])
    labels=np.array([0,1,1,0], dtype=np.uint8)
    classes=[0,1]
    features = (features-np.mean(features,axis=0))/np.std(features,axis=0) #normalization
    #f) Train the network for 200 epochs and a batch size of 1
    history=model_a.fit(features, labels,
              batch_size=1,
              epochs=400,
              verbose=0) #1 or 0 (Show each Epoch toggle)
    
    """
    #Evaluate (Evaluate, Plot)
    test_samples=np.array([[0,0],[1,1], [1,0]])
    test_y=np.array([0,0,1])
    score=model_a.evaluate(test_samples,test_y)
    print('\nTotal loss on testing set: ', score[0])
    print('Accuracy of testing set: ', score[1], "\n")
    
    #Predict (Classes, Probability)
    test_samples=np.array([[0,-2],[2,5]])
    test_class1_prob=model_a.predict(test_samples)
    print('\nThe probability of class 1 for the test samples is: \n', 
          test_class1_prob)
    test_lab=np.uint8(test_class1_prob>0.5)
    print('The classes for the test samples is: \n', test_lab, "\n")
    """
    
    #weights
    weights=model_a.layers[0].get_weights() 
    plt.figure(figsize=[15,5])
    plt.subplot(1,3,1)
    for node_i in range(weights[0].shape[1]):
      thre_parms=np.array(weights[0][:,node_i])#This first item is the weights for the inputs 
      thre_parms=np.append(thre_parms, weights[1][node_i]) #second item the weights for the bias
      plot_fun_thr(features,labels,thre_parms,classes)
    #plt.show()
    
    #curve
    #h) Plot the training loss (i.e., the learning curve) for all the epochs.
    plt.subplot(1,3,2)
    acc_curve=np.array(history.history['accuracy'])
    loss_curve=np.array(history.history['loss'])
    plot_curve(acc_curve,loss_curve)
    plt.show()
    
    """
    j) What behavior do you observe from the classifier lines after adding more nodes?
    Which number of nodes is more suitable in this problem? Explain. 
    
    After adding more nodes to the first layer it is evident the network has a smoother
    learning curve on average with the random weights and 100% accuracy is more common.
    It is also evident the extra nodes make it possible to individualize each quadrant
    of points. With the current data-set it would be more practical and efficient 
    to utilize 2 nodes in the first layer. 2-4 would be acceptable but > 4 nodes
    would be excessive. 
    """

"""
Problem 2) [Python] Application of Keras to build, compile, and train a neural network as a
three-class classifier for MNIST dataset (0 vs. 1 vs. 2):
""" 
def p_2(layers, L1_nodes, L2_nodes):
    
    def img_plt(images, labels):
      plt.figure()
      for i in range(1,11):
        plt.subplot(2,5,i)
        plt.imshow(images[i-1,:,:],cmap='gray')
        plt.title('Label: ' + str(labels[i-1]))
      plt.show()
    
    def feat_extract(images):
      width=images.shape[1]
      height=images.shape[2]
      features=np.zeros((images.shape[0],4))
      features_temp=np.sum(images[:,0:int(width/2),0:int(height/2)],axis=2)#quadrant 0
      features[:,0]=np.sum(features_temp,axis=1)/(width*height/4)
      features_temp=np.sum(images[:,0:int(width/2),int(height/2):],axis=2) #quadrant 1
      features[:,1]=np.sum(features_temp,axis=1)/(width*height/4)
      features_temp=np.sum(images[:,int(width/2):,0:int(height/2)],axis=2) #quadrant 2
      features[:,2]=np.sum(features_temp,axis=1)/(width*height/4)
      features_temp=np.sum(images[:,int(width/2):,int(height/2):],axis=2)  #quadrant 3
      features[:,3]=np.sum(features_temp,axis=1)/(width*height/4)
      return features
    
    def feat_plot(features,labels,classes):
      for class_i in classes:
        plt.plot(features[labels[:]==classes[class_i],0], 
                 features[labels[:]==classes[class_i],1],'o', markersize=15)
      #plt.axis([-2,2,-2,2])
      plt.xlabel('x: feature 1')
      plt.ylabel('y: feature 2')
      plt.legend(['Class'+str(classes[class_i]) for class_i in classes])
      plt.show()
    
    def acc_fun(labels_actual, labels_pred):
      acc=np.sum(labels_actual==labels_pred)/len(labels_actual)*100
      return acc

    """
    a) Use mnist function in keras.datasets to load MNIST dataset and split it into training
    and testing sets. Then, randomly select 20% of the training images along with their
    corresponding labels to be the validation data.
    """
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    #Selecting only 0 and 1 digits from the training and testing sets
    classes=[0,1,2]
    x_train_012=x_train[np.logical_or.reduce((y_train==0,y_train==1,y_train==2)),0:28,0:28]
    y_train_012=y_train[np.logical_or.reduce((y_train==0,y_train==1,y_train==2))]
    #print('Samples of the training images')
    #img_plt(x_train_012[0:10,:,:],y_train_012[0:10])
    
    x_test_012=x_test[np.logical_or.reduce((y_test==0,y_test==1,y_test==2)),0:28,0:28]
    y_test_012=y_test[np.logical_or.reduce((y_test==0,y_test==1,y_test==2))]
    #print('Samples of the testing images')
    #img_plt(x_test_012[0:10,:,:],y_test_012[0:10])
    
    #shuffling training data
    num_train_img=x_train_012.shape[0]
    #print(num_train_img)
    train_ind=np.arange(0,num_train_img)
    train_ind_s=np.random.permutation(train_ind)
    x_train_012=x_train_012[train_ind_s,:,:]
    y_train_012=y_train_012[train_ind_s]
     
    #Choose the percent to split by
    split_per = 20
    split_per = split_per/100

    #select % of train data as validation
    x_val_012 = x_train_012[0:int(split_per*num_train_img),:,:]
    y_val_012 = y_train_012[0:int(split_per*num_train_img)]
    #print('Samples of the validation images')
    #img_plt(x_valid[0:10,:,:],y_valid[0:10])
    
    #the rest of the train data 
    x_train_012 = x_train_012[int(split_per*num_train_img):,:,:]
    y_train_012 = y_train_012[int(split_per*num_train_img):]
    
    """
    print("# of images in x_train:", x_train.shape[0], "\n# of images in y_train (labels):", y_train.shape[0])
    print("\nx_train image shape:", x_train.shape[1:], "\ny_train (labels) image shape:", y_train.shape[1:])
    print("\n# of images in x_test:", x_test.shape[0], "\n# of images in y_test (labels):", y_test.shape[0])
    print("\nx_test image shape:", x_test.shape[1:], "\ny_test (labels) image shape:", y_test.shape[1:])
    print("\n# of images in x_valid:", x_valid.shape[0], "\n# of images in y_valid (labels):", y_valid.shape[0])
    print("\nx_valid image shape:", x_valid.shape[1:], "\ny_valid (labels) image shape:", y_valid.shape[1:])
    """

    """Calculating the training, validation and testing feature (average of the four quadrants grid)
    b) Feature extraction: average the pixel values in the quadrants in each image to
    generate a feature vector of 4 values for each image.
    """
    feature_train=feat_extract(x_train_012)
    feature_val=feat_extract(x_val_012)
    feature_test=feat_extract(x_test_012)
    
    """
    c) Convert the label vectors for all the sets to binary class matrices using
    to_categorical() Keras function.
    """
    y_train_012_c = to_categorical(y_train_012, len(classes))
    y_val_012_c = to_categorical(y_val_012, len(classes))
    y_test_012_c = to_categorical(y_test_012, len(classes))
    
    #d) Build, compile, train, and then evaluate:
    #Build a neural network with 1 layer that contains 10 nodes using Keras
    model_a=Sequential()
    if layers == 1:
        model_a.add(Dense(input_dim=4, units=L1_nodes, activation='tanh'))
    if layers == 2:
        model_a.add(Dense(input_dim=4, units=L1_nodes, activation='tanh'))
        model_a.add(Dense(units=L2_nodes, activation='tanh'))
    model_a.add(Dense(units=3, activation='softmax'))
    #model_a.summary()
    """Compile (optimizer, loss, metrics)
    Compile the network. Make sure to select a correct loss function for this
    classification problem. Use stochastic gradient descent learning (SGD,
    learning rate of 0.0001). Explain your selection of the loss function. 
    
    I selected categorical for loss function because output will be 0, 1 or 2.
    """
    opt = tf.keras.optimizers.SGD(learning_rate=0.0001)
    model_a.compile(loss='categorical_crossentropy',
                    optimizer=opt,
                    metrics=['accuracy'])
    #Fit (Batch Size, epochs)
    #Train the network for 50 epochs and a batch size of 16.
    history=model_a.fit(feature_train, y_train_012_c,
                        batch_size=16,
                        epochs=50,
                        verbose=0) #1 or 0 (Show each Epoch toggle)
    
    print("**********\nLayers:", layers, "\nLayer 1 nodes:",
          L1_nodes, "\nLayer 2 nodes:", L2_nodes, "\n**********")
    #Evaluating the model on the training samples
    score=model_a.evaluate(feature_train,y_train_012_c)
    print('Total loss on training set: ', score[0])
    print('Accuracy of training set: ', score[1])
    
    #Evaluating the model on the validation samples
    score=model_a.evaluate(feature_val,y_val_012_c)
    print('Total loss on validation set: ', score[0])
    print('Accuracy of validation set: ', score[1])
    
    plt.figure(figsize=[9,5])
    acc_curve=np.array(history.history['accuracy'])
    loss_curve=np.array(history.history['loss'])
    plot_curve(acc_curve,loss_curve)

    """
    f) 
    What behavior do you observe in the training loss and the validation loss when you
    increase the number layers and nodes in the previous table. Which model is more
    suitable in this problem? Explain. 
    
    Upon increasing the number of layers and nodes it is evident the training
    and validation loss decreases when more nodes are added. In model 4 by
    adding another layer with an additional 10 nodes the loss increases meaning
    it was not significantly beneficial to add layers unless the additional layer
    has sufficient nodes like in model 5. 
    Therefore, I believe model 3 is the most suitable. model 5 has the best results
    however it may be computationally excessive.
    """
    
    """
    g) Evaluate the selected model in part (e) on the testing set and report the
    testing loss and accuracy.
    
    For model 3 
    testing loss = 0.3320
    testing accuracy = 0.8773
    
    For model 5 
    testing loss = 0.3214
    testing accuracy = 0.8776
    """
    score=model_a.evaluate(feature_test,y_test_012_c)
    print('Total loss on testing set: ', score[0])
    print('Accuracy of testing set: ', score[1])
    
"""
Problem 3) [Python] Application of Keras to build, compile, and train a neural network to
classify songs from Spotify dataset.
""" 
    
def p_3(): #expect 85,86,87
    import sklearn
    from sklearn.model_selection import train_test_split
    
    import pandas as pd

    #spotify = pd.read_csv (r'')
    #print (spotify)
    
    #from google.colab import files
    
    #uploaded = files.upload()

    spotify = pd.read_csv('DeepLearning\Data\spotify_preprocessed.csv')

    print(spotify)
    x_train, x_test = train_test_split(spotify, test_size=0.10, shuffle=True)

    

def main():
    p_1()
    p_2(1, 10, 0)
    #p_2(1, 50, 0)
    #p_2(1, 100, 0)
    #p_2(2, 100, 10)
    #p_2(2, 100, 50)
    p_3()

print("-----------------------------------------------------")
    
if (__name__=="__main__"):
    
    main()


print("-----------------------------------------------------")     
     