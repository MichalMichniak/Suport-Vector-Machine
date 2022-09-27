import random
import svm
import numpy as np
import matplotlib.pyplot as plt

def main():
    x = [[np.random.uniform()*10 - 5,np.random.uniform()*10 -5] for i in range(100)] # np.random.uniform()
    x1 = [[np.random.uniform()*10,np.random.uniform()*10] for i in range(100)]
    y = [1 for i in range(100)]
    y1 = [-1 for i in range(100)]
    x = np.array(x+x1)
    y = np.array(y+y1)
    n_ = [i for i in range(200)]
    random.shuffle(n_)
    #plt.plot(x[:,0],x[:,1],".b")
    lr = svm.krenel_1D_SVM(1/100,2,0.00001)
    for k in range(100):
        n_ = [i for i in range(200)]
        for n,i in enumerate(n_):
            lr.teach(x[i],y[i])
    for n,i in enumerate(x):
        print(lr.predict(i))
        if lr.predict(i):
            plt.plot(i[0],i[1],".r")
        else:
            plt.plot(i[0],i[1],".b")
    plt.show()
    pass

if __name__ == "__main__":
    main()