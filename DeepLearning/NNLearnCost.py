# -*- coding: utf-8 -*-
"""
Introduction to Deep Learning – CAP 4613
Assignment 4
Problem 2
Due: Sunday, March 20 before 11:59 pm

Spyder (Python 3.8) IDE

link to final version in Colab:
https://colab.research.google.com/drive/1iiaDWGffHniZTsJi33SWS7a7DmNphlc8?usp=sharing
"""
import webbrowser
import matplotlib.pyplot as plt
import numpy as np

#url = 'https://colab.research.google.com/drive/1iiaDWGffHniZTsJi33SWS7a7DmNphlc8?usp=sharing'
#webbrowser.open(url)  # Go to the google colab

#part a) Create class NeuralNetwork():
class NeuralNetwork(object):
    
    """ part a.i)
    initializes a 3x1 weight vector randomly and 
    initializes the learning rate to 1. 
    """
    def __init__(self, num_params, learning_r):  
        #create random weight
        #np.random.seed(1)
        #self.weight_matrix = 2 * np.random.random((num_params, 1)) - 1
        self.weight_matrix = np.ones(num_params)
        #history variable that saves the weights and the training cost after each epoch
        self.weight_cost_hist = np.zeros(num_params + 1)
        self.l_rate = learning_r
        #self.act_fun = 0 #activation function limit
    
    """ part a.ii)
    performs the forward propagation by multiplying the inputs by the neuron
    weights and then generating the output. 
    
    **done in training function**
    """
    def forward_propagation(self, inputs): 
        #print("input:\n", inputs.shape)
        #print("weights:\n", self.weight_matrix.shape)
        v = np.dot(inputs, self.weight_matrix)
        return self.hard_limiter(v)
    
    """ part iii)
    performs the perceptron learning rule for num_train_iterations 
    times using the inputs and labels. 
    """
    def train(self, train_inputs, labels, num_train_epochs = 10):
        # Number of iterations we want to perform for this set of input.
        for iteration in range(num_train_epochs):
            #updating the perceptron base on the misclassified examples
            #print("\n*****************************  Iteration #: ", iteration + 1)
            adjust_val = np.zeros(train_inputs[0].shape[0])
            error_sqr_hist = 0
            cost = 0
            
            #go through each individual input
            for i in range(train_inputs.shape[0]):
                #print("Current Input", train_inputs[i])
                #print("Current Weight: \n", self.weight_matrix)
                y = np.dot(train_inputs[i], self.weight_matrix)
                #print("y: ", y)
                
                # Calculate the error in the output.
                error = labels[i] - y
                error_sqr_hist += error*error
                #print("error: ", error, "\n")
                #print("error_sqr_hist: ", error_sqr_hist, "\n")
                # Find (d-y)(x0 x1 x2)
                adjust = np.multiply(error, train_inputs[i])
                adjust_val = np.vstack([adjust_val, adjust])
                
            #print("Adjusted x values: \n", adjust_val[1:], "\n")
            #print("column sums: ", np.sum(adjust_val[1:], axis = 0))
            
            d_weight = np.sum(adjust_val[1:]*(self.l_rate/train_inputs.shape[0]), axis = 0)
            #print("\nd_weight: ", d_weight)
            d_weight = np.add(d_weight, self.weight_matrix)
            #print("adjusted_d_weight: ", d_weight)
            self.weight_matrix = d_weight
            cost = float(1/(2*train_inputs.shape[0]))*error_sqr_hist
            #print("cost: ", cost)
            
            new_wc = np.append(d_weight, cost)
            #print("\nnew_wc: ", new_wc)
            self.weight_cost_hist = np.vstack([self.weight_cost_hist, new_wc])
            #print("weight_cost_history: \n", self.weight_cost_hist[1:])
    
        
    def plot_fun_thr(self, features, labels, thre_parms, classes):
        #ploting the data points
        x1_n1 = []
        x2_n1 = []
        x1_1 = []
        x2_1 = []
        
        for x in range(len(features)):
            if labels[x] == 1:
                x1_1.append(features[x][1])
                x2_1.append(features[x][2])
            if labels[x] == -1 or labels[x] == 0:
                x1_n1.append(features[x][1])
                x2_n1.append(features[x][2])
         
        plt.figure(figsize=(6, 10))
        plt.suptitle("GDL PLOTS")
        plt.subplot(2,1,1)
        plt.title("Data and Classifier")
#part c)Plot the given data points with two different markers for each group.
        plt.plot(x1_1, x2_1, 'b^', label = 'Class: ' + str(classes[1]))
        plt.plot(x1_n1, x2_n1, 'r*', label = 'Class: ' + str(classes[0]))
                
        #find graph bounds
        x1_min = np.amin(features, axis=0)[1] #x1_min
        x1_max = np.amax(features, axis=0)[1] #x1_max
        x2_min = np.amin(features, axis=0)[2] #x2_min
        x2_max = np.amax(features, axis=0)[2] #x2_max
        plt.axis([x1_min - 2, x1_max + 2, x2_min - 2, x2_max + 2])
        
#Part d) Use the trained weights at each epoch and plot the classifier lines of every 5 epochs
#(i.e., 1,5,10, ..., 50) 
        epoch = 1
        x0 = self.weight_cost_hist[1][0]
        x1 = self.weight_cost_hist[1][1]
        x2 = self.weight_cost_hist[1][2]

        x_val = np.linspace(-4, 4, 100)
        y_val = -(x1*x_val+x2)/x0
        plt.plot(x_val, y_val, color = 'r', label = "Training Classifier")
        
        for i in range(0, len(self.weight_cost_hist), 5):
            if i == 0:
                pass
            else:
                epoch = i
                x0 = self.weight_cost_hist[i][0]
                x1 = self.weight_cost_hist[i][1]
                x2 = self.weight_cost_hist[i][2]
    
                x_val = np.linspace(-4, 4, 100)
                y_val = -(x1*x_val+x2)/x0
                plt.plot(x_val, y_val, color = 'r', alpha = 0.2)  
        
#part e) trained weights and plot the final classifier line to the plot in (c) 
        #aX1 + bX2 + c-0 --> x2 = -(aX1 + c)/b
        x1 = np.linspace(x1_min - 2, x1_max + 2, 100)
        x2 = -(thre_parms[1]*x1+thre_parms[2])/thre_parms[0] 
        plt.plot(x1, x2, color = 'g', label='Trained Classifier')
        plt.legend()
        
        #defining and showing plot
        plt.xlabel('x1: (feature 1)')
        plt.ylabel('x2: (feature 2)')
        plt.show() 
        
#part f) Plot the training cost (i.e., the learning curve) for all the epochs    
        plt.subplot(2,1,2)
        
        epoch = []
        cost = []
        for i in range(1,len(self.weight_cost_hist)):
            epoch.append(i)
            cost.append(self.weight_cost_hist[i][3])
            
        #find graph bounds
        x_min = min(epoch) #x_min
        x_max = max(epoch) #x_max
        y_min = 0          #y_min
        y_max = max(cost)  #y_max
        
        plt.title("Epoch and Cost")
        plt.axis([x_min, x_max, y_min, y_max+y_max/4])
        plt.xlabel('epoch')
        plt.ylabel('cost') 
        plt.plot(epoch, cost)
        plt.show() 
        
def plot_3_lr(L1, L2, L3):
    plt.suptitle("Epoch and Cost")
    plt.subplot(1,3,1)
    epoch = []
    cost = []
    for i in range(1,len(L1)):
        epoch.append(i)
        cost.append(L1[i][3])
    plt.plot(epoch, cost, label = "Learning Curve")
    plt.legend()
    plt.title("Learning Rate: 1")
    plt.axis([min(epoch), max(epoch), 0, max(cost)])
    plt.ylabel('Cost') 
    
    plt.subplot(1,3,2)
    epoch = []
    cost = []
    for i in range(1,len(L2)):
        epoch.append(i)
        cost.append(L2[i][3])
    plt.plot(epoch, cost, label = "Learning Curve")
    plt.legend()
    plt.title("Learning Rate: 0.5")
    plt.axis([min(epoch), max(epoch), 0, max(cost)])
    plt.xlabel('Epoch')
    
    plt.subplot(1,3,3)
    epoch = []
    cost = []
    for i in range(1,len(L3)):
        epoch.append(i)
        cost.append(L3[i][3])
    plt.plot(epoch, cost, label = "Learning Curve")
    plt.legend()
    plt.title("Learning Rate: 0.05")
    plt.axis([min(epoch), max(epoch), 0, max(cost)])
 
    plt.show()     
             
        
def main():

    """Testing on the example in number 1

    data = np.array([ [1,1], [1,0], [0,1], [-1,-1], [-1,0], [-1,1] ])
    #add bias (1) to arrary
    features = np.insert(data, 0, 1, axis = 1)
    print("Inputs: \n", features)
    labels = np.array([1,1,0,0,0,0])
    print("\nLabels: \n", labels)
    classes = [0,1]
    
    print("\nClasses: \n", classes)
    R1 = NeuralNetwork(features[0].size, 0.1)
    print("\n-------Train-------\n")
    print("Initial weights:", R1.weight_matrix, "\nLearning Rate: ", R1.l_rate)
    R1.train(features, labels, 50)
    print("\n\nTrained Weight and Cost history:\n",R1.weight_cost_hist[1:], "\n")
    
    # plot the seperating line based on the weights
    R1.plot_fun_thr(features, labels, R1.weight_matrix, classes)
    
    """

#part b)Use the gradient descent rule to train a single neuron on the datapoints given:
    #part b.i) Create an np array of a shape 10x2 that contains the inputs
    test_data = np.array([ [1,1], [1,0], [0,1], [-1,-1], [0.5,3],
                         [0.7,2], [-1,0], [-1,1], [2,0], [-2,-1] ])
    #part b.i) Create an np array of a shape 10x1 that contains the labels
    test_label = np.array([1,1,-1,-1,1,1,-1,-1,1,-1])
    #part b.ii) Add the bias to the inputs array to have a 10x3 shape. 
    test_data_b = np.insert(test_data, 0, 1, axis = 1)
    classes = [-1,1]
    
    #part b.iii) Create the network with one neuron using the class NeuralNetwork() 
    # with learning rate of 1 then train it using train(inputs, labels, 50) function.
    L1 = NeuralNetwork(test_data_b[0].size, 0.05)
    L1.train(test_data_b, test_label, 100)
    
    #plot the data and classifier based on the trained weights
    L1.plot_fun_thr(test_data_b, test_label, L1.weight_matrix, classes)
    
#part g) Repeat step (b) with the learning rates of 0.5 and 0.05. Create a subplot with the 
#learning curves of learning rates 1, 0.5, and 0.05. Add titles. 
    L2 = NeuralNetwork(test_data_b[0].size, 0.5)
    L2.train(test_data_b, test_label, 50)
    
    L3 = NeuralNetwork(test_data_b[0].size, 0.05)
    L3.train(test_data_b, test_label, 50)
    
    plot_3_lr(L1.weight_cost_hist, L2.weight_cost_hist, L3.weight_cost_hist)
    
    """
    I can see that with a high learning rate like "1" the learning curve undergoes drastic 
    changes to the point it gets higher with each epoch. At learning rate of "0.5"  the
    curve is able to reache a low cost quite fast however it does so by drastically
    dropping immediately. Upon lowering the learning curve to "0.05" the curve smoothes
    out and the learning curve undergoes less drastic changes. For this data set I believe 
    having a learning curve of "0.05" is the best.
    """

print("-----------------------------------------------------")
    
if (__name__=="__main__"):
    
    main()

print("-----------------------------------------------------")        
        
        
        
        
        
        
        
        
        