import numpy as np
class FullyConnectedLayer:
	def __init__(self, in_nodes, out_nodes):
		# Method to initialize a Fully Connected Layer
		# Parameters
		# in_nodes - number of input nodes of this layer
		# out_nodes - number of output nodes of this layer
		self.in_nodes = in_nodes
		self.out_nodes = out_nodes
		# Stores the outgoing summation of weights * feautres 
		self.data = None

		# Initializes the Weights and Biases using a Normal Distribution with Mean 0 and Standard Deviation 0.1
		self.weights = np.random.normal(0,0.1,(in_nodes, out_nodes))	
		self.biases = np.random.normal(0,0.1, (1, out_nodes))
		self.input = []
		###############################################
		# NOTE: You must NOT change the above code but you can add extra variables if necessary 

	def forwardpass(self, X):
		# print('Forward FC ',self.weights.shape)
		# Input
		# activations : Activations from previous layer/input
		# Output
		# activations : Activations after one forward pass through this layer
		
		n = X.shape[0]  # batch size
		# INPUT activation matrix  		:[n X self.in_nodes]
		# OUTPUT activation matrix		:[n X self.out_nodes]

		###############################################

		output = np.zeros((n,self.out_nodes))
		output = output + np.dot(X,self.weights)
		output = output + self.biases[0,:]
		self.input = sigmoid(output)
		return sigmoid(output)
		###############################################
		
	def backwardpass(self, lr, activation_prev, delta):
		# Input
		# lr : learning rate of the neural network
		# activation_prev : Activations from previous layer
		# delta : del_Error/ del_activation_curr
		# Output
		# new_delta : del_Error/ del_activation_prev
		
		# Update self.weights and self.biases for this layer by backpropagation
		n = activation_prev.shape[0] # batch size

		###############################################
		# TASK 2 - YOUR CODE HERE
		g_dash_later = self.input*(1-self.input)
		new_delta = np.transpose(np.dot(self.weights,np.transpose(delta*g_dash_later)))

		bias_activation = np.ones(activation_prev.shape)
		del_Error = np.dot(np.transpose(activation_prev),delta*g_dash_later)
		bias_error = np.sum(delta*g_dash_later,axis=0)
		self.weights = self.weights - lr*del_Error	
		self.biases = self.biases - lr*bias_error

		return new_delta 
		###############################################

class ConvolutionLayer:
	def __init__(self, in_channels, filter_size, numfilters, stride):
		# Method to initialize a Convolution Layer
		# Parameters
		# in_channels - list of 3 elements denoting size of input for convolution layer
		# filter_size - list of 2 elements denoting size of kernel weights for convolution layer
		# numfilters  - number of feature maps (denoting output depth)
		# stride	  - stride to used during convolution forward pass
		self.in_depth, self.in_row, self.in_col = in_channels
		self.filter_row, self.filter_col = filter_size
		self.stride = stride

		self.out_depth = numfilters
		self.out_row = int((self.in_row - self.filter_row)/self.stride + 1)
		self.out_col = int((self.in_col - self.filter_col)/self.stride + 1)

		# Stores the outgoing summation of weights * feautres 
		self.data = None
		
		# Initializes the Weights and Biases using a Normal Distribution with Mean 0 and Standard Deviation 0.1
		self.weights = np.random.normal(0,0.1, (self.out_depth, self.in_depth, self.filter_row, self.filter_col))	
		self.biases = np.random.normal(0,0.1,self.out_depth)
		

	def forwardpass(self, X):
		# print('Forward CN ',self.weights.shape)
		# Input
		# X : Activations from previous layer/input
		# Output
		# activations : Activations after one forward pass through this layer
		n = X.shape[0]  # batch size
		# INPUT activation matrix  		:[n X self.in_depth X self.in_row X self.in_col]
		# OUTPUT activation matrix		:[n X self.out_depth X self.out_row X self.out_col]

		###############################################
		# TASK 1 - YOUR CODE HERE
		output = np.zeros((n,self.out_depth,self.out_row,self.out_col))
	

		# for p in range(n):
		# 	for d in range(self.out_depth):
		# 		for i in range(self.out_row):
		# 			for j in range(self.out_col):
		# 				output[p,d,i,j] = np.sum(X[p,:,i*self.stride : i*self.stride+self.filter_row, j*self.stride : j*self.stride+self.filter_col]*self.weights[d,:,:,:])

		# 		output[p,d,:,:] = output[p,d,:,:] + self.biases[d]
		broad_weights = np.repeat(np.reshape(self.weights,(1,self.out_depth, self.in_depth, self.filter_row, self.filter_col)),n,axis=0)
		broad_biases = np.repeat(np.repeat(np.repeat(np.reshape(self.biases,(1,self.out_depth,1,1)),n,axis=0),self.out_row,axis=2),self.out_col,axis=3)
		for i in range(self.out_row):
			for j in range(self.out_col):
				inp = X[:,:,i*self.stride : i*self.stride+self.filter_row, j*self.stride : j*self.stride+self.filter_col]
				broad_inp = np.repeat(np.reshape(inp,(n,1,self.in_depth,self.filter_row,self.filter_col)),self.out_depth,axis=1)
				output[:,:,i,j] = np.sum(np.sum(np.sum(broad_weights*broad_inp,axis=4),axis=3),axis=2)

		output = output + broad_biases
		self.data = output
		return sigmoid(output)
		###############################################

	def backwardpass(self, lr, activation_prev, delta):
		# Input
		# lr : learning rate of the neural network
		# activation_prev : Activations from previous layer
		# delta : del_Error/ del_activation_curr
		# Output
		# new_delta : del_Error/ del_activation_prev
		
		# Update self.weights and self.biases for this layer by backpropagation
		n = activation_prev.shape[0] # batch size

		###############################################
		# TASK 2 - YOUR CODE HERE
		del_Error = np.zeros((n,self.out_depth, self.in_depth, self.filter_row, self.filter_col))
		pre_calc = np.repeat(np.reshape(delta*derivative_sigmoid(self.data),(n,self.out_depth,1,self.out_row,self.out_col)),self.in_depth,axis=2)
		del_bias = np.zeros((n,self.out_depth))
		
		for i in range(self.filter_row):
			for j in range(self.filter_col):
				sub_inp = activation_prev[:,:,i:i+self.stride*self.out_row:self.stride,j:j+self.stride*self.out_col:self.stride]
				#print(i,j,sub_inp.shape,self.in_row,self.in_col,self.stride)
				broad_inp = np.repeat(np.reshape(sub_inp,(n,1,self.in_depth,self.out_row,self.out_col)),self.out_depth,axis=1)
				del_Error[:,:,:,i,j] = np.sum(np.sum(pre_calc*broad_inp,axis=4),axis=3)

		del_bias = np.sum(np.sum(delta*derivative_sigmoid(self.data),axis=3),axis=2)

		new_delta = np.zeros((n,self.in_depth,self.in_row,self.in_col))

		g_dash_delta = delta*derivative_sigmoid(self.data)

		broad_weights = np.repeat(np.reshape(self.weights,(1,self.out_depth, self.in_depth, self.filter_row, self.filter_col)),n,axis=0)


		for p in range(self.in_row):
			for q in range(self.in_col):
				new_delta[:,:,p,q] = np.dot(np.sum(np.sum(delta*derivative_sigmoid(self.data),axis=3),axis=2),self.weights[:,:,p%self.stride,q%self.stride])

		self.weights = self.weights - lr*np.sum(del_Error,axis=0)		
		self.biases = self.biases - lr*np.sum(del_bias,axis=0)		

		return new_delta


		###############################################
	
class AvgPoolingLayer:
	def __init__(self, in_channels, filter_size, stride):
		# Method to initialize a Convolution Layer
		# Parameters
		# in_channels - list of 3 elements denoting size of input for max_pooling layer
		# filter_size - list of 2 elements denoting size of kernel weights for convolution layer

		# NOTE: Here we assume filter_size = stride
		# And we will ensure self.filter_size[0] = self.filter_size[1]
		self.in_depth, self.in_row, self.in_col = in_channels
		self.filter_row, self.filter_col = filter_size
		self.stride = stride

		self.out_depth = self.in_depth
		self.out_row = int((self.in_row - self.filter_row)/self.stride + 1)
		self.out_col = int((self.in_col - self.filter_col)/self.stride + 1)

	def forwardpass(self, X):
		# print('Forward MP ')
		# Input
		# X : Activations from previous layer/input
		# Output
		# activations : Activations after one forward pass through this layer
		
		n = X.shape[0]  # batch size
		# INPUT activation matrix  		:[n X self.in_depth X self.in_row X self.in_col]
		# OUTPUT activation matrix		:[n X self.out_depth X self.out_row X self.out_col]

		###############################################
		# TASK 1 - YOUR CODE HERE
		output = np.zeros((n,self.out_depth,self.out_row,self.out_col))
		# for p in range(n):
		# 	for d in range(self.out_depth):
		# 		for i in range(self.out_row):
		# 			for j in range(self.out_col):
		# 				output[p,d,i,j] = np.sum(X[p,d,i*self.stride : i*self.stride+self.filter_row, j*self.stride : j*self.stride+self.filter_col]/(self.filter_row*self.filter_col))

		for i in range(self.out_row):
			for j in range(self.out_col):
				inp = X[:,:,i*self.stride : i*self.stride+self.filter_row, j*self.stride : j*self.stride+self.filter_col]
				output[:,:,i,j] = np.sum(np.sum(inp,axis=3),axis=2)/(self.filter_col*self.filter_row) 
		return output
		###############################################


	def backwardpass(self, alpha, activation_prev, delta):
		# Input
		# lr : learning rate of the neural network
		# activation_prev : Activations from previous layer
		# activations_curr : Activations of current layer
		# delta : del_Error/ del_activation_curr
		# Output
		# new_delta : del_Error/ del_activation_prev
		
		n = activation_prev.shape[0] # batch size

		###############################################
		# TASK 2 - YOUR CODE HERE
		new_delta = np.ones((n,self.in_depth,self.in_row,self.in_col))
		for i in range(self.in_row):
			for j in range(self.in_col):
				p  = delta[:,:,int(i/self.stride),int(j/self.stride)]
				new_delta[:,:,i,j] = p*1/(self.filter_col*self.filter_row)
		return new_delta
		###############################################


# Helper layer to insert between convolution and fully connected layers
class FlattenLayer:
    def __init__(self):
        pass
    
    def forwardpass(self, X):
        self.in_batch, self.r, self.c, self.k = X.shape
        return X.reshape(self.in_batch, self.r * self.c * self.k)

    def backwardpass(self, lr, activation_prev, delta):
        return delta.reshape(self.in_batch, self.r, self.c, self.k)


# Helper Function for the activation and its derivative
def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def derivative_sigmoid(x):
	return sigmoid(x) * (1 - sigmoid(x))
