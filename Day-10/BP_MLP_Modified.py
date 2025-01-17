import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Raw data
raw_data = '''
8.56   -1.22   4.67   -0.00   -0.15   -0.10   37.79   4.15   1
8.57   -1.24   5.00   0.02   -0.08   -0.04   37.79   4.15   1
8.77   -1.13   4.76   0.02   -0.12   -0.03   37.87   4.15   1
8.71   -0.98   4.86   0.06   -0.14   -0.03   37.79   4.15   1
8.66   -1.00   5.11   0.08   -0.05   -0.06   37.85   4.14   1
8.79   -1.34   4.83   0.05   -0.08   -0.10   37.87   4.14   1
8.09   -1.00   5.13   0.14   -0.04   -0.10   37.79   4.13   1
8.53   -1.23   4.72   0.01   0.11   -0.16   37.83   4.15   1
7.84   -3.50   5.12   -0.04   -0.02   -0.27   37.93   4.12   2
8.41   -2.76   4.53   0.22   0.25   -0.31   37.95   4.11   2
7.81   -3.39   6.08   0.16   -0.30   -0.01   37.95   4.11   2
6.93   -3.36   4.38   -0.02   0.33   0.36   37.99   4.13   2
8.43   -3.18   4.58   -0.23   -0.02   0.03   37.93   4.12   2
7.91   -3.67   5.27   -0.07   -0.70   -0.10   37.93   4.13   2
7.89   -2.84   5.16   0.10   -0.09   -0.27   37.95   4.11   2
8.98   -3.66   4.36   -0.08   0.14   -0.30   38.01   4.13   2
7.87   -2.94   4.66   0.26   -0.11   -0.11   38.01   4.12   2
8.20   -4.08   4.52   0.08   -0.14   -0.19   38.03   4.11   2
8.21   -3.21   5.36   -0.05   0.11   0.06   37.99   4.13   2
8.24   -3.34   4.37   -0.00   0.49   0.29   38.01   4.13   2
7.43   -3.15   5.38   -0.13   -0.18   0.06   37.95   4.13   2
7.34   -3.29   5.75   0.23   -0.17   -0.15   37.93   4.13   2
7.93   -3.72   5.89   -0.03   -0.10   -0.43   37.93   4.12   2
8.75   -3.86   4.75   -0.00   -0.12   -0.08   37.95   4.11   2
8.62   0.15   4.81   0.03   -0.15   -0.14   37.19   4.15   3
8.57   -0.02   5.03   0.14   -0.12   -0.08   37.27   4.14   3
8.35   0.15   5.12   0.10   -0.06   0.02   37.25   4.14   3
8.34   0.11   5.18   0.04   -0.11   -0.02   37.19   4.15   3
8.65   -0.04   5.02   0.01   -0.13   -0.03   37.21   4.15   3
8.47   -0.16   5.18   -0.02   -0.08   -0.02   37.19   4.14   3
8.80   -0.05   5.15   0.09   -0.03   -0.10   37.21   4.15   3
8.57   -0.34   5.27   -0.05   -0.30   -0.09   37.21   4.13   3
8.59   0.05   5.16   0.07   -0.02   -0.07   37.21   4.15   3
8.57   0.06   4.91   -0.07   0.02   -0.13   37.25   4.16   3
'''

# Parse the raw data into a numpy array
data = np.array([list(map(float, line.split())) for line in raw_data.strip().split('\n')])

#split faetures and labels
x=data[:,:-1]#features(all cloumn except last)
y=data[:,-1]

#one-hot encoding the labels
def one_hot_encode(y,num_classes):
    return np.eye(num_classes)[y.astype(int)-1]

y_encoded=one_hot_encode(y,3)

#split into training and testing sets
x_train,x_test,y_train,y_test=train_test_split(x,y_encoded,test_size=0.2,random_state=42)

#normalize the data
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

#MLP with Backpropagation
class MLP:
    def _init_(self,input_size,hidden_size,output_size,learning_rate=0.01):
        #initialize weights and biases
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.output_size=output_size
        self.learning_rate=learning_rate

        #weights initialization
        self.W1=np.random.randn(input_size,hidden_size)*0.1
        self.b1=np.zeros(hidden_size)
        self.W2=np.random(hidden_size,output_size)*0.1
        self.b2=np.zeros(output_size)

    def sigmoid (self,z):
        return 1/(1+np.exp(-z))
    
    def sigmoid_derivative(self,z):
        return z*(1-z)
    
    def forward(self, X):
    # Forward pass
        self.z1 = np.dot(X, self.W1)+ self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2= self.sigmoid(self.z2)
        return self.a2

    def backward(self, X, y):
        # Backward pass (Gradient computation)
        m = X.shape[0]

        # Output layer error
        self.output_error = self.a2 - y
        self.output_delta = self.output_error * self.sigmoid_derivative(self.a2)
        
        self.hidden_error = self.output_delta.dot(self.W2.T)
        self.hidden_delta = self.output_error * self.sigmoid_derivative(self.a1)
        
        self.W1_grad = X.T.dot(self.hidden_delta)
        self.b1_grad = np.sum(self.hidden_delta, axix = 0)
        self.W2_grad = self.a1.T.dot(self.hidden_delta)
        self.b2_grad = np.sum(self.output_delta, axix = 0)
    
    def update_weights(self):
         
         self.W1 -= self.learning_rate * self.W1_grad
         self.b1 -= self.learning_rate * self.b1_grad
         self.W2 -= self.learning_rate * self.W2_grad
         self.b2 -= self.learning_rate * self.b2_grad

    def train(self, X, y, epochs=1000):
        errors = []
        for epoch in range(epochs):
            self.forward(X)
            self.backward(X, y)
            self.update_weights()
            # Compute error
            loss = np.mean(np.square(self.a2 - y))
            errors.append(loss)

        return errors
    
