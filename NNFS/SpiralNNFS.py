import numpy as np
import matplotlib.pyplot as plt
from nnfs.datasets import spiral_data
import nnfs

nnfs.init() # does three things to get random seed set to 0 by default , 
            # creates a dtype of 32 bit float int and overrides og dot product from numpy
X,y = spiral_data(samples=100,classes=3)

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        # Initialize weights and biases
        self.weights=0.01*np.random.randn(n_inputs,n_neurons)
        self.biases=np.zeros((1,n_neurons))
        
    # Forward pass
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        # Calculate output values from inputs, weights and biases
        
dense1 = Layer_Dense(2, 3)
# Perform a forward pass of our training data through this layer
dense1.forward(X)
# Let's see output of the first few samples:
print(dense1.output[:5]) 

plt.scatter(X[:,0],X[:,1],c=y,cmap='brg')
plt.show()