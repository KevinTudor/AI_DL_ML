# -*- coding: utf-8 -*-
"""
Introduction to Deep Learning – CAP 4613
Assignment 3
Due: Sunday, February 20 before 11:59 pm

Name: Kevin Tudor z-number:Z23468207

Spyder (Python 3.8) IDE

link to final version in Colab:
https://colab.research.google.com/drive/1aDylUnQ_Aa-ZZt9vbQ5y7dvLIWKhsHn3?usp=sharing
"""
import webbrowser
import matplotlib.pyplot as plt
import numpy as np

#url = 'https://colab.research.google.com/drive/1aDylUnQ_Aa-ZZt9vbQ5y7dvLIWKhsHn3?usp=sharing'
#webbrowser.open(url)  # Go to the google colab

class NeuralNetwork(object):
    
    """
    initializes a 3x1 weight vector randomly and 
    initializes the learning rate to 1. 
    """
    def __init__(self, num_params = 3):  
        
        self.weight_matrix = 2 * np.random((num_params, 1)) - 1
        self.l_rate = 1
    
    """
    performs the hard-limiter activation on the nx1 vector x. 
    """
    def hard_limiter(self, x):
        outs = np.zeros(x.shape)
        outs[x>0] = 1
        return outs
    
    """
    performs the forward propagation by multiplying the inputs by the neuron
    weights and passing the output through the hard_limiter activation function. 
    """
    def forward_propagation(self, inputs): 
        outs = np.dot(inputs, self.weight_matrix)
        return self.hard_limiter(outs)
    
    """
    performs the perceptron learning rule for num_train_iterations 
    times using the inputs and labels. 
    """
    #def train(self, inputs, labels, num_train_iterations=10): 
    def train(self, train_inputs, train_outputs,num_train_iterations = 1000):
        # Number of iterations we want to perform for this set of input.
        for iteration in range(num_train_iterations):
            #updating the perceptron base on the misclassified examples
            for i in range(train_inputs.shape[0]):
                pred_i = self.pred(train_inputs[i,:])
                if pred_i != train_outputs[i]:
                    output = self.forward_propagation(train_inputs[i,:])
                    # Calculate the error in the output.
                    error = train_outputs[i] - output
                    adjustment = self.l_rate*error*train_inputs[i]
                    # Adjust the weight matrix
                    self.weight_matrix[:,0] += adjustment
                    # plot the seperating line based on the weights
                    print('Iteration #' + str(iteration))
                    NeuralNetwork.plot_fun_thr(train_inputs[:,0:2], train_outputs, 
                                               self.weight_matrix[:,0])
    
    """
    classifies the inputs to either class 0 or 1 by multiplying 
    them by the neuron weights, passing the output through the hard_limiter 
    activation function and thresholding. 
    """
    def pred(self,inputs): 
        prob = self.forward_propagation(inputs)
        preds = np.int8(prob >= 0.5)
        return preds
        
        
    def plot_fun(features, labels, classes):
        plt.plot(features[labels[:]==classes[0],0], features[labels[:]==classes[0],1], 'rs',
                 features[labels[:]==classes[1],0], features[labels[:]==classes[1],1], 'g^')
        plt.axis([-2,2,-2,2])
        plt.xlabel('x: feature 1')
        plt.ylabel('y: feature 2')
        plt.legend(['Class'+str(classes[0]), 'Class'+str(classes[1])])
        #plt.pause(0.5)
        plt.show()    
        
        
    def plot_fun_thr(features, labels, thre_parms, classes):
    #ploting the data points
        plt.plot(features[labels[:]==classes[0],0], features[labels[:]==classes[0],1], 'rs',
                 features[labels[:]==classes[1],0], features[labels[:]==classes[1],1], 'g^')
        plt.axis([-1,2,-1,2])
        #ploting the seperating line
        x1 = np.linspace(-1,2,50)
        #aX1 + bX2 + c-0 --> x2 = -(aX1 + c)/b
        x2 = -(thre_parms[0]*x1+thre_parms[2])/thre_parms[1] 
        plt.plot (x1, x2, '-r')
        plt.xlabel('x: feature 1')
        plt.ylabel('y: feature 2')
        plt.legend(['Class'+str(classes[0]), 'Class'+str(classes[1])])
        #plt.pause(0.5)
        plt.show()    
        
        
        
def main():
    features = np.array([[0,0], [0,1], [1,0], [1,1]])
    print(features)
    labels = np.array([0,0,0,1])
    print(labels)
    classes = [0,1]
    
    NeuralNetwork.plot_fun(features, labels, classes)

print("-----------------------------------------------------")
    
if (__name__=="__main__"):
    
    main()


print("-----------------------------------------------------")        
        
        
        
        
        
        
        
        
        