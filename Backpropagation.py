#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!pip install opencv-python
import enum
import numpy as np 
import pandas as pd
import os
from PIL import Image


# In[2]:


def readData(path : str):
    dataset = []
    testData = []
    
    dataset = pd.read_csv(path)

    class_ = dataset.iloc[:, -1].unique()

    dataset['lable'] = -1
    for i in range(len(class_)):
        dataset.loc[(dataset.iloc[:, -2] == class_[i]), 'lable'] = i


    colun = dataset.columns
    testData = dataset.iloc[:20, :]
    testData = testData.append(dataset.iloc[50:70, :])
    testData = testData.append(dataset.iloc[100:120, :])

    trainData = dataset.iloc[20:50, :]
    trainData = testData.append(dataset.iloc[70:100, :])
    trainData = testData.append(dataset.iloc[120:, :])

    trainData = trainData.sample(frac=1)
    trainY = trainData['lable']
    trainData = trainData.drop([colun[-2], 'lable'], axis=1)


    trainData = np.array(trainData)
    trainY = np.array(trainY).reshape(-1, 1)

    return trainData, trainY, testData.drop([colun[-2], 'lable'], axis=1).values, testData['lable'].values.reshape(-1,1)


# In[3]:


class helperFunction:
    @staticmethod
    def one_hot_encoding(y, n_classes):
        classes = np.arange(0, n_classes).reshape(1, n_classes)

        encoded_y = (y == classes).astype('float128')

        return encoded_y
    
    @staticmethod
    def intializeWeights(l1, l2):
        np.random.seed(20)
        return np.random.randn(l1, l2).astype('float128') * 1e-2
    
    @staticmethod
    def intializeBais(l2):
        return np.zeros((1, l2)).astype('float128')
    
    @staticmethod
    def tanh(z):
        ep = np.exp(-2 * z)
        res = (1 - ep) / (1 + ep)
        return res
    
    @staticmethod
    def Dtanh(z):
        act = np.square(helperFunction.tanh(z))
        return (1 - act)
    
    @staticmethod
    def sigmoid(z):
        return 1/(1 + np.exp(-z))
    
    @staticmethod
    def Dsigmoid(z):
        sig = helperFunction.sigmoid(z)
        return sig * (1 - sig)
    
    @staticmethod
    def Loss_fun(Y, y):
        err = Y * np.log(helperFunction.clip(Y)) + (1.0 - helperFunction.clip(Y)) * np.log(1.0 - helperFunction.clip(y))
        err *= (1.0 / Y.shape[0])
        err = np.sum(-err)
        return err

    @staticmethod
    def signalError(Y, y):
        indexer = y.argmax(axis=-1)

        y = np.zeros_like(y)
        # print(indexer.squeeze().shape)
        y[:,indexer] = 1

        return (Y - y)

    @staticmethod
    def clip(a):
      
        maxval = 1.0 - 1e-8
        minval = 1e-8

        return np.clip(a, a_min=minval, a_max=maxval)


# In[4]:


class activation1(enum.Enum):
    tanh = helperFunction.tanh
    sigmoid = helperFunction.sigmoid
class DNN:
    def __init__(self, X:np.array, Y:np.array, Layers:list, activ:list, lr = 0.0001, epoch = 10, bais = True, batchSize = 1, costFunction = 1):
        self.batchSize = batchSize
        self.X = X
        self.Y = helperFunction.one_hot_encoding(Y, len(np.unique(Y)))
        self.L_dict = {}
        self.DL_dict = {}
        self.epoch = epoch
        self.lr = lr
        self.bais = bais
        self.Layers = Layers
        self.activ = activ
        self.out = self.Y
        self.costFunction = costFunction
        for l in range(1, len(Layers)):
            self.L_dict['W' + str(l)] = helperFunction.intializeWeights(Layers[l - 1], Layers[l])
            self.L_dict['b' + str(l)] = helperFunction.intializeBais(Layers[l])
                
        
            
    def feedForward(self):
        for l in range(1, len(self.Layers) - 1):
            self.L_dict['z' + str(l)] = np.dot(self.L_dict['a' + str(l - 1)], self.L_dict['W' + str(l)]) + self.L_dict['b' + str(l)]
            self.L_dict['a' + str(l)] = self.activ[l - 1](self.L_dict['z' + str(l)])
            
        L = len(self.Layers) - 1
        self.L_dict['z' + str(L)] = np.dot(self.L_dict['a' + str(L - 1)], self.L_dict['W' + str(L)]) + self.L_dict['b' + str(L)]
        
        self.L_dict['a' + str(L)] = self.activ[L - 1](self.L_dict['z' + str(L)])
        
    def backWard(self, numOfiteration):
        L = len(self.Layers) - 1
        if self.costFunction == 1:
            self.DL_dict['z' + str(L)] = (1 / self.L_dict['a0'].shape[0]) * helperFunction.signalError(self.L_dict['a' + str(L)], self.out)
        else:
            self.DL_dict['z' + str(L)] = (1 / self.L_dict['a0'].shape[0]) * (self.L_dict['a' + str(L)] - self.out)
        
        for l in reversed(range(1, L + 1)):
            self.DL_dict['W' + str(l)] = np.dot(self.L_dict['a' + str(l - 1)].T, self.DL_dict['z' + str(l)])
            if self.bais:
                self.DL_dict['b' + str(l)] = np.sum(self.DL_dict['z' + str(l)],axis=0, keepdims=True)
            
            if l != 1:
                self.DL_dict['a' + str(l - 1)] = np.dot(self.DL_dict['z' + str(l)], self.L_dict['W' + str(l)].T)
                self.DL_dict['z' + str(l - 1)] = self.DL_dict['a' + str(l - 1)] * (helperFunction.Dtanh(self.L_dict['z' + str(l - 1)]) if self.activ[l - 1] == helperFunction.tanh else helperFunction.sigmoid(self.L_dict['z' + str(l - 1)]))
                
        for l in reversed(range(1, L + 1)):
            self.L_dict['W' + str(l)] -= ((self.lr/(numOfiteration + 1)) * self.DL_dict['W' + str(l)])
            if self.bais:
                self.L_dict['b' + str(l)] -= ((self.lr /(numOfiteration + 1))* self.DL_dict['b' + str(l)])
            
    def fit(self):
        for i in range(1, self.epoch + 2):
            for example in range(0, self.X.shape[0], self.batchSize):
                self.L_dict['a0'] = self.X[example: example + self.batchSize, :]
                self.out = self.Y[example: example + self.batchSize, :].reshape(-1, self.Y.shape[1])
                self.feedForward()
                self.backWard(i//100)
            prediction = self.predict(self.X)
            if i % 100 == 0:
                print(helperFunction.Loss_fun(self.Y, prediction))
            
    def predict(self, X):
        self.L_dict['a0'] = X
        self.feedForward()
        return self.L_dict['a' + str(len(self.Layers) - 1)]


# In[5]:


# data = readData('/media/elfeky/7C30-7ED1/College/4th/CI/Task2/Task_CI/T19/IrisData.txt', 0)
# data = data.sample(frac=1)
#
#
# # In[6]:
#
#
# class_  = data['Class'].unique()
#
#
# # In[7]:
#
#
# data['lable'] = -1
# for i in range(len(class_)):
#     data.loc[(data['Class'] == class_[i]),'lable'] = i
#
#
# # In[8]:
#
#
# trainData = data.drop(['Class', 'lable'], axis=1)
# trainY = data['lable']
#
#
# # In[9]:
#
#
# trainData = np.array(trainData)
# trainY = np.array(trainY).reshape(-1, 1)
#
#
# # In[10]:
#
#
# model = DNN(trainData, trainY, [4, 7, 3], [activation1.tanh, activation1.sigmoid], epoch=500, lr=5e-2, batchSize=1)
#
#
# # In[11]:
#
#
# model.fit()
#
