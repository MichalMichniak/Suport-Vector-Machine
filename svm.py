import numpy as np

class linear_SVM:
    def __init__(self,Lambda,n,alpha = 0.01) -> None:
        """
        args:
            Lambda : float - constant (usualy inverse of epoch number)
            n : int - lenght of w vector
        """
        self.lambda_ = Lambda
        self.w = np.array([1 for i in range(n)],dtype=float)
        self.b = 1
        self.alpha = alpha
        pass

    def predict(self, x):
        return np.sign(self.w.T@x - self.b) >0

    def teach(self,x : np.ndarray,y : int):
        t = self.w.T @ x
        if y*(t) < 1:
            for n,i in enumerate(self.w):
                self.w[n] -= self.alpha*(2*self.lambda_*self.w[n] - x[n]*y)
                self.b += self.alpha*y
            pass
        else:
            for n,i in enumerate(self.w):
                self.w[n] -= self.alpha*(2*self.lambda_*self.w[n])
            pass

class krenel_1D_SVM:
    def __init__(self,Lambda,n,alpha = 0.01, krenel_func = lambda x: np.exp(-1*(x[0]**2+x[1]**2))) -> None:
        """
        args:
            Lambda : float - constant (usualy inverse of epoch number)
            n : int - lenght of w vector
        """
        self.lambda_ = Lambda
        self.w = np.array([1 for i in range(n+1)],dtype=float)
        self.b = 1
        self.alpha = alpha
        self.krenel_func = krenel_func
        pass

    def krenel(self, v):
        """
        return new vector with bigger dimensionality
        """
        return np.array(list(v) + [self.krenel_func(v)])

    def predict(self, x):
        """
        predict if x in class
        """
        x = self.krenel(x)
        return np.sign(self.w.T@x - self.b) >0

    

    def teach(self,x : np.ndarray,y : int):
        """
        teach model with new vector
        """
        x = self.krenel(x)
        t = self.w.T @ x
        if y*(t) < 1:
            for n,i in enumerate(self.w):
                self.w[n] -= self.alpha*(2*self.lambda_*self.w[n] - x[n]*y)
                self.b += self.alpha*y
            pass
        else:
            for n,i in enumerate(self.w):
                self.w[n] -= self.alpha*(2*self.lambda_*self.w[n])
            pass