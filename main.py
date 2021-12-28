import tkinter.messagebox
from tkinter import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Backpropagation import *


root = Tk()
root.title('Back propagation')
root.geometry("1300x800")
root.configure(background='blue')

# global variables any value here just as a initialization not the actual
numOfhidden = int()
numOfneurons = int()
lr = int()
epochs = int()
datasetchoice = int()
activationchoice = int()
baisChoice = bool()
istrained = bool()
Model = list()
# confusion
con = tuple()
# cost function
costFunc = 0
testDataX = list()
testDataY = list()
def collectData():
    global numOfhidden
    global numOfneurons
    global lr
    global epochs
    global datasetchoice
    global activationchoice
    global baisChoice, costFunc

    numOfhidden = int(HiddenLayers.get())
    numOfneurons = list(map(int, NofNeurons.get().split(',')))
    print(type(numOfneurons), NofNeurons.get().split())
    lr = float(learningRate.get())
    epochs = int(NofEpochs.get())
    activationchoice = activation.get()
    datasetchoice = dataset.get()
    baisChoice = Bias.get()
    costFunc = errorFunction.get()
def trainModel():
    collectData()
    global Model, numOfneurons, istrained
    global testDataX
    global testDataY
    global testDataX, testDataY, costFunc

    X = []
    Y = []
    if datasetchoice == 1:
        mnistData= pd.read_csv('mnist_train.csv')
        X = mnistData.iloc[:,1:].values
        Y = mnistData.iloc[:,0].values.reshape(-1,1)

    else:
        X, Y, testDataX, testDataY = readData('IrisData.txt')

    print("shapes ",X.shape, Y.shape)
    numOfneurons.insert(0, X.shape[1])
    numOfneurons.append(len(np.unique(Y)))
    actvList = list()
    for i in range(numOfhidden + costFunc):
        if activationchoice == 1:
            actvList.append(activation1.tanh)
        else:
            actvList.append(activation1.sigmoid)
    if not bool(costFunc):
        actvList.append(activation1.sigmoid)
    print(numOfneurons, actvList, lr, epochs, baisChoice == 1, costFunc)
    Model = DNN(X=X, Y=Y, Layers=numOfneurons, activ=actvList, lr=lr, epoch=epochs, bais=(baisChoice == 1), batchSize=1,costFunction= costFunc)
    Model.fit()
    istrained = True
    tkinter.messagebox.showinfo(root, "Done!")

def testModel():
    global Model, testDataX, testDataY, con
    X = 0
    Y = 0
    if not istrained:
        tkinter.messagebox.showinfo(root, "Please train the model first")
        return
    if datasetchoice:
        mnistData = pd.read_csv('mnist_test.csv')
        X = mnistData.iloc[:, 1:]
        Y = mnistData.iloc[:, 0].values.reshape(-1, 1)
    else:
        X = testDataX
        Y = testDataY
    y = Model.predict(X)
    acc = (Y == y.argmax(axis=1).reshape(-1, 1)).sum()/len(Y)
    tkinter.messagebox.showinfo(root, f"accuracy {acc}")

    con = conf(Y.squeeze(), y.argmax(axis=1).reshape(-1, 1).squeeze())

def confMatrix():
    global con
    if not istrained:
        tkinter.messagebox.showinfo(root, "Please train the model first")
        return

    drawMatrix(con[0], con[1])

def conf(Y:np.array, y:np.array):
    cls = len(np.unique(Y))
    cons = np.zeros((cls, cls))
    print(Y.shape, y.shape)
    for i in range(len(Y)):
        cons[Y[i]][y[i]] += 1

    return cons, cls


def drawMatrix(cons, cls):
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.matshow(cons, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(cons.shape[0]):
        for j in range(cons.shape[1]):
            ax.text(x=j, y=i, s=int(cons[i, j]), va='center', ha='center', size='xx-large')

    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    plt.show()


# block of input
def makeBlock(text:str, text2:str, root:Tk, rows:int):

    l1 = Label(root, text=text, font=('helvetica', 20), relief=GROOVE, bd=1, fg="#00ff00", bg='blue')
    l1.pack(pady=20)
    l1.place(x=10, y=rows)

    inputField = Entry(root, width=30)
    inputField.pack()
    inputField.place(x=len(text) * 15, y=rows + 10)

    if text2:
        l2 = Label(root, text=text2, font=('helvetica', 10))
        l2.pack(pady=20)
        l2.place(x=600 + len(text) * 7, y=rows + 10)
    return inputField

# control the rows
rows = 60

HiddenLayers = makeBlock('Enter Number of Hidden Layers', None, root, 10)

NofNeurons = makeBlock('Enter Number of neurons at each hidden layer ', "separate with (,) e.g 4, 8, 3", root, rows)

learningRate = makeBlock('Enter Learning Rate ', None, root, rows * 2)

NofEpochs = makeBlock('Enter Number of Epochs ', None, root, rows * 3)

activation = IntVar(root)
dataset = IntVar(root)
errorFunction = IntVar(root)


sigmoidRdio = Radiobutton(root, text='Sigmoid', variable=activation, value=0)

sigmoidRdio.place(x=10, y=rows * 4)
TanhRdio = Radiobutton(root, text='Hyperbolic Tangent', variable=activation, value=1)

TanhRdio.place(x=200, y=rows * 4)



Bias = IntVar()
Bias_checkBox = Checkbutton(root, text='Bias', variable=Bias, onvalue=1, offvalue=0)
Bias_checkBox.place(x=10, y=rows * 5)



trainButton = Button(root, text="Train", command=trainModel)
trainButton.place(x=10, y=rows * 7)
testButton = Button(root, text="Test", command=testModel)
testButton.place(x=200, y=rows * 7)

confusionmatrixButton = Button(root, text="Calculate confusion matrix", command=confMatrix)
confusionmatrixButton.pack()
confusionmatrixButton.place(x=400, y=rows * 7)

IrisDatasetRadio1 = Radiobutton(root, text='Iris', variable=dataset, value=0)
IrisDatasetRadio1.pack()
IrisDatasetRadio1.place(x=10, y=rows * 6)
mnistDatasetRadio2 = Radiobutton(root, text='Mnist', variable=dataset, value=1)
mnistDatasetRadio2.pack()
mnistDatasetRadio2.place(x=200, y=rows * 6)

IrisDatasetRadio1 = Radiobutton(root, text='Cross Entropy', variable=errorFunction, value=0)
IrisDatasetRadio1.pack()
IrisDatasetRadio1.place(x=10, y=rows * 8)
mnistDatasetRadio2 = Radiobutton(root, text='signal error', variable=errorFunction, value=1)
mnistDatasetRadio2.pack()
mnistDatasetRadio2.place(x=200, y=rows * 8)






if __name__ == "__main__":
    root.mainloop()



