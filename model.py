import numpy as np
import os

class ThreeLayerNN:
    def __init__(self, input_size, hidden1, hidden2, output_size, 
                 activation='relu', reg_lambda=0.01):
        self.params = {}
        # He初始化
        self.params['W1'] = np.random.randn(input_size, hidden1) * np.sqrt(2.0/input_size)
        self.params['b1'] = np.zeros(hidden1)
        self.params['W2'] = np.random.randn(hidden1, hidden2) * np.sqrt(2.0/hidden1)
        self.params['b2'] = np.zeros(hidden2)
        self.params['W3'] = np.random.randn(hidden2, output_size) * np.sqrt(2.0/hidden2)
        self.params['b3'] = np.zeros(output_size)
        
        self.activation = activation
        self.reg_lambda = reg_lambda
    
    def forward(self, X):
        z1 = np.dot(X, self.params['W1']) + self.params['b1']
        a1 = self._activate(z1)
        
        z2 = np.dot(a1, self.params['W2']) + self.params['b2']
        a2 = self._activate(z2)
        
        z3 = np.dot(a2, self.params['W3']) + self.params['b3']
        exp_scores = np.exp(z3 - np.max(z3, axis=1, keepdims=True))
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        
        cache = {'z1':z1, 'a1':a1, 'z2':z2, 'a2':a2, 'probs':probs}
        return probs, cache
    
    def backward(self, X, y, cache):
        m = X.shape[0]
        a1, a2 = cache['a1'], cache['a2']
        probs = cache['probs']
        
        dz3 = probs.copy()
        dz3[range(m), y] -= 1
        dz3 /= m
        
        dW3 = np.dot(a2.T, dz3) + self.reg_lambda * self.params['W3']
        db3 = np.sum(dz3, axis=0)
        
        da2 = np.dot(dz3, self.params['W3'].T)
        dz2 = da2 * self._activate_deriv(cache['z2'])
        dW2 = np.dot(a1.T, dz2) + self.reg_lambda * self.params['W2']
        db2 = np.sum(dz2, axis=0)
        
        da1 = np.dot(dz2, self.params['W2'].T)
        dz1 = da1 * self._activate_deriv(cache['z1'])
        dW1 = np.dot(X.T, dz1) + self.reg_lambda * self.params['W1']
        db1 = np.sum(dz1, axis=0)
        
        return {'dW1':dW1, 'db1':db1, 'dW2':dW2, 'db2':db2, 'dW3':dW3, 'db3':db3}
    
    def _activate(self, z):
        if self.activation == 'relu':
            return np.maximum(0, z)
        elif self.activation == 'sigmoid':
            return 1/(1+np.exp(-z))
        elif self.activation == 'tanh':
            return np.tanh(z)
    
    def _activate_deriv(self, z):
        if self.activation == 'relu':
            return (z > 0).astype(float)
        elif self.activation == 'sigmoid':
            s = 1/(1+np.exp(-z))
            return s*(1-s)
        elif self.activation == 'tanh':
            return 1 - np.tanh(z)**2
    
    def save(self, path):
        np.savez(path, **self.params)
    
    def load(self, path):
        params = np.load(path)
        for key in self.params:
            self.params[key] = params[key]