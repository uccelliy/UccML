import numpy as np
import pandas as pd
import kernel as Kernel
import optimizer as Optimizer

class SVM:
    def __init__(self, Data,C=1.0, kernel=Kernel.rbf_kernel, lr=0.001, epochs=1000):
        self.C = C           
        self.kernel_func = kernel
        self.lr = lr
        self.epochs = epochs
        self.alphas = None
        self.b = 0
        self.X = Data.iloc[:,1:-1].values
        self.y = Data.iloc[:,-1].values.astype(float)
    
    def fit(self):
        n_samples = self.X.shape[0]
        self.alphas = np.zeros(n_samples)
        self.b = 0
        params = {'alphas': self.alphas, 'b': self.b}
        params = Optimizer.gd_optimizer(params, svm_grad_loss, 
                                        lr=self.lr, epochs=self.epochs, X=self.X, y=self.y,C=self.C,kernel=self.kernel_func)
        self.alphas = np.clip(params['alphas'], 0, self.C)
        self.b = params['b']
    
    def _decision_function(self, x):
        result = sum(self.alphas[i] * self.y[i] * self.kernel_func(self.X[i], x) 
                     for i in range(len(self.X)))
        return result + self.b
    
    def predict(self, X):
        return np.sign(np.array([self._decision_function(x) for x in X]))
    


def svm_grad_loss(params, X, y,C,kernel_func):
    alphas, b = params['alphas'], params['b']
    n_samples = X.shape[0]
    grad_alpha = np.zeros_like(alphas)
    grad_b = 0
    for i in range(n_samples):
        f_i = sum(alphas[j] * y[j] * kernel_func(X[j], X[i]) for j in range(n_samples)) + b
        grad_alpha[i] = 1 - y[i] * f_i 
        grad_b += -y[i] * (1 if 1 - y[i]*f_i > 0 else 0)     
    return {'alphas': grad_alpha, 'b': grad_b / n_samples}
    