from numpy import *  
import operator  
import csv,logging,time  
import logging.config  
  
def loadTrainData():  
    ''''' 
    Load the train data from train.csv,and split label and data. 
    '''  
    I = []  
    with open('E:\\kaggle\\train.csv','rb') as file:  
        lines = csv.reader(file)  
        for line in lines:  
            I.append(line) #42001*785  
    I.remove(I[0]) # Remove the describe row  
    I = array(I)   # Array the train data  
    label = I[:,0] #42000*1,Get the label column  
    data = I[:,1:] #42000*784, Get the data block  
    return normalizing(toInt(data)),toInt(label)  
  
def loadTestData():  
    ''''' 
    Load the test data from test.csv,and cut the description. 
    '''  
    I = []  
    with open('E:\\kaggle\\test.csv','rb') as file:  
        lines = csv.reader(file)  
        for line in lines:  
            I.append(line) #28001*784  
    I.remove(I[0]) #remove description  
    array_I = array(I) #28000*784  
    return normalizing(toInt(array_I))  
  
def toInt(array):  
    ''''' 
    Exchange the elements'type of array as int type from str. 
    '''  
    array=mat(array)     
    rows,lines = shape(array)  
    newArray = zeros((rows,lines))  
    for i in xrange(rows):  
        for j in xrange(lines):  
            newArray[i,j] = int(array[i,j])  
    return newArray  
  
def normalizing(array):  
    ''''' 
    Normalizing the elements of input array.All the values normalizing 0 or 1(!=0) 
    '''  
    rows,lines = shape(array)  
    for i in xrange(rows):  
        for j in xrange(lines):  
            if array[i,j]!=0:  
                array[i,j]=1  
    return array  

def classify(inX,dataSet,labels,k):  
    ''''' 
    Classifying by K-NN algorithm 
    '''  
    inX = mat(inX)  
    dataSet = mat(dataSet)  
    labels = mat(labels)  
    dataSetSize = dataSet.shape[0]  
    diffMat = tile(inX,(dataSetSize,1)) - dataSet # Make a diff between train data and test data  
    sqDiffMat = array(diffMat)**2  # Make square for diffMat  
    sqDistance = sqDiffMat.sum(axis=1) # Sum by row  
    distance = sqDistance**0.5  
    sortedDistIndecies = distance.argsort()  
    classCount={}  
    for i in xrange(k):  
        votellabel = labels[0,sortedDistIndecies[i]]  
        classCount[votellabel] = classCount.get(votellabel,0)+1  
    sortedClassCount = sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)  
    return sortedClassCount[0][0]  