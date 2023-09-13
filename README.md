# AI_DL_ML
Artificial Intelligence, Deep Learning, and Machine Learning projects

## Deep Learning

### Technology

* [Keras](https://keras.io/): Open-source library that provides a Python interface for artificial neural networks. Keras acts as an interface for the TensorFlow library.
* [TensorFlow](https://www.tensorflow.org/): Free and open-source software library for machine learning and artificial intelligence. It can be used across a range of tasks but has a particular focus on training and inference of deep neural networks
* [MNIST dataset](https://www.tensorflow.org/datasets/catalog/mnist): The MNIST database is a large database of handwritten digits that is commonly used for training various image processing systems. The database is also widely used for training and testing in the field of machine learning. It was created by "re-mixing" the samples from NIST's original datasets. The MNIST dataset is divided into two sets - training and test. Each set comprises a series of images (28 x 28-pixel images of handwritten digits) and their respective labels (values from 0 - 9, representing which digit the image corresponds to).
* [Matplotlib](https://matplotlib.org/): a plotting library for the Python programming language and its numerical mathematics extension NumPy. It provides an object-oriented API for embedding plots.
* [Numpy](https://numpy.org/doc/stable/user/whatisnumpy.html): a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays.

### [KerasTensorFlowSpotify](DeepLearning/KerasTensorFlowSpotify.py) using [Preprocessed Spotify Dataset](DeepLearning/Data/spotify_preprocessed.csv) - [Preview](DeepLearning/Output/KerasTensorFlowSpotify_P3Spotify.png)
- P1) Employ the Keras library to create a versatile neural network that spans the entire workflow – from building the model, configuring features, and compiling, to training, evaluation, and making classifications based on learned weights.
- P2) Leverage Keras to model, compile, train and validate a neural network tailored as a three-class classifier for the MNIST dataset, adept at distinguishing between the digits 0, 1, and 2.
- P3) Utilize Keras to model, compile, train and validate a specialized neural network for the task of classifying songs within the Spotify dataset (based on preprocessed features of each song), highlighting the chance of the song being in the weekly top 100 charts.

[Output:](DeepLearning/Output) [Final Classifier Line, Accuracy, and Training Loss](DeepLearning/Output/KerasTensorFlowSpotify_ClassifierLine.png), [Neural Network Details and Accuracy](DeepLearning/Output/KerasTensorFlowSpotify_P2NN.png), and [Spotify Top 100 Prediction Model Accuracy](DeepLearning/Output/KerasTensorFlowSpotify_P3SpotifyValidation.png)

### [Keras_MNIST](DeepLearning/Keras_MNIST.py)
- Create a figure with ten subplots, each corresponding to digits 0 through 9 from the MNIST dataset. Display the respective digit as the subplot title. Then, proceed to split the MNIST dataset into training and testing sets and provide essential details, including the number of images in each set and their image dimensions. Next, randomly select 20% of the training images and their corresponding labels, assigning them to x_valid and y_valid for validation of the training set. Ultimately, implement a loop to invoke the plot function for images from the validation set to classify each number and plot them in ascending order.

[Output:](DeepLearning/Output) [MNIST Validation Plot](DeepLearning/Output/Keras_MNIST_Validation.png)

### [NNLearnCost](DeepLearning/NNLearnCost.py)
- The primary objective of this program is to train a classifier capable of distinguishing between two distinct classes and subsequently report the associated learning cost. Begin by initializing a 3x1 weight vector randomly and set the learning rate to 1. Proceed with forward propagation by multiplying the inputs with the neuron weights to generate the output. In the data and classifier chart, you'll notice a thick red line representing the initial classifier line. Additionally, less prominent red lines depict the classifier's evolution at each epoch. Finally, the green line signifies the classifier's state after completion of the training process.

[Output:](DeepLearning/Output) [Classifier Line Initial, Training and Validation](DeepLearning/Output/NNLearnCost_ClassifierLine.png)

### [NNPropogation](DeepLearning/NNPropogation.py)
- Conduct the forward propagation process by multiplying the inputs with the neuron weights and passing the resulting output through the hard_limiter activation function. Subsequently, apply the perceptron learning rule for a specified number of training iterations, using both the input data and corresponding labels. The ultimate goal is to classify inputs into either class 0 or 1. To achieve this classification, multiply the inputs by the neuron weights, pass the output through the hard_limiter activation function once more, and then apply thresholding to make the final classification decision.

### [SimplePythonCalc](DeepLearning/SimplePythonCalc.py)
- In Python, implement a simple calculator that does the following operations: summation, subtraction, multiplication, division, mod, power, exp, natural log, and abs. 


## Machine Learning


