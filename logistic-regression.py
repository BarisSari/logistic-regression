'''  Developed by Bayram Baris Sari
*   E-mail: bayrambariss@gmail.com
*   Tel No: +90 539 593 7501    
*
*   This is the implementation of Logistic Regression
*   Pseudo-code is taken from "Introduction to 
*   Machine Learning"
*   
*   Iris dataset is used as input. 10-fold cross
*   validation is applied for testing.
'''

import numpy as np
import csv
from random import uniform
from random import shuffle

max_iter = 200
learning_rate = 0.1


def split(dat):
    # splitting data to two parts: features and expected results
    features = []
    expected = []
    for i in range(len(dat)):
        features.append(dat[i][0:4])
        expected.append(dat[i][4])
    return features, expected


def denominator(scores):
    # this computes the denominator part of softmax function
    # since np.exp(709) is the limit for overflow, I first checked it and then put the values to np.exp
    for i in range(3):
        if scores[i] > 709:
            scores[i] = 709
    return np.exp(scores[0])+np.exp(scores[1])+np.exp(scores[2])


def find_class(string):
    # it find r vector for the class
    if string == "Iris-setosa":
        return [1, 0, 0]
    elif string == "Iris-versicolor":
        return [0, 1, 0]
    else:
        return [0, 0, 1]


def biggest(a, b, c):
    # finding biggest probability. A means class-0, B is class-1 and C is class-2
    i = 'Iris-setosa'
    Max = a
    if b > Max:
        Max = b
        i = 'Iris-versicolor'
    if c > Max:
        i = 'Iris-virginica'

    return i


def train(N, K, d, features, expected):
    w = np.zeros((K, d))    # weight vector
    gradient_w = np.zeros((K, d))  # gradient vector
    o = np.zeros((3, 1))  # observed output
    y = np.zeros((3, 1))  # probability for each class

    # Weight vectors are randomly chosen at the beginning
    for i in range(K):
        for j in range(d):
            w[i][j] = uniform(-0.01, 0.01)

    iteration = 0
    while iteration < max_iter:
        # Gradient vectors are set to 0
        for i in range(K):
            for j in range(d):
                gradient_w[i][j] = 0

        # for each sample,t, in training data
        for t in range(N):
            r = find_class(expected[t])
            # observed output is set to 0
            for i in range(K):
                o[i] = 0
                # and observed output is calculated py W[i]*features[t]
                for j in range(d):
                    o[i] += np.dot(w[i][j], np.transpose(features[t][j]))
            # computing the probability for each class for sample,t
            for i in range(K):
                prev_y = y.copy()
                # for avoiding overflow, if o[i] is set to 709 if its bigger than 709
                if o[i] > 709:
                    o[i] = 709
                y[i] = np.exp(o[i]) / denominator(o)
                # if a nonnumeric value returned from equation above, use the previous probability
                if np.isnan(y).any():
                    y = prev_y
            # compute gradient
            for i in range(K):
                for j in range(d):
                    gradient_w[i][j] += (np.dot(r[i] - y[i], np.transpose(features[t][j])))
        # update weight vector
        for i in range(K):
            for j in range(d):
                w[i][j] += learning_rate * gradient_w[i][j]

        iteration += 1
    return w


def accuracy(pre, exp):
    correct = 0
    wrong = 0
    for i in range(len(pre)):
        if pre[i] == exp[i]:
            correct += 1
        else:
            wrong += 1
    acc = correct / (correct+wrong)
    print("Accuracy: (%d/%d)" % (int(correct), int(correct+wrong)),"{0:.3f}%".format((correct/(correct+wrong))*100))
    return acc


def confusion(matrix, pre, actual):
    ''' Confusion matrix =
                                                    Predicted
                                  Iris-setosa    Iris-versicolor    Iris-virginica
                Iris-setosa         [0][0]           [0][1]             [0][2]
     Actual     Iris-versicolor     [1][0]           [1][1]             [1][2]
                Iris-virginica      [2][0]           [2][1]             [2][2]

    '''
    if pre == actual == 'Iris-setosa':
        matrix[0][0] += 1
    elif actual == 'Iris-setosa' and pre == 'Iris-versicolor':
        matrix[0][1] += 1
    elif actual == 'Iris-setosa' and pre == 'Iris-virginica':
        matrix[0][2] += 1

    elif actual == 'Iris-versicolor' and pre == 'Iris-setosa':
        matrix[1][0] += 1
    elif pre == actual == 'Iris-versicolor':
        matrix[1][1] += 1
    elif actual == 'Iris-versicolor' and pre == 'Iris-virginica':
        matrix[1][2] += 1

    elif actual == 'Iris-virginica' and pre == 'Iris-setosa':
        matrix[2][0] += 1
    elif actual == 'Iris-virginica' and pre == 'Iris-versicolor':
        matrix[2][1] += 1
    elif pre == actual == 'Iris-virginica':
        matrix[2][2] += 1

    return matrix


K = 3  # number of classes
d = 4  # number of features in each sample
data = []
acc_ratio = []
confusion_matrix = np.zeros((3, 3))

text_file = open("irisdata.csv", "r")
text_file.readline()
test = csv.reader(text_file, csv.QUOTE_NONNUMERIC, delimiter=',')
for row in test:
    if row:
        data.append([float(row[0]), float(row[1]), float(row[2]), float(row[3]), row[4]])
text_file.close()

# shuffle is necessary for 10-fold cross validation
shuffle(data)

'''
There are 150 samples in total and we will split it as 135-15 samples for 10-fold cross-validation 
Each time,Training data will have 135 samples and test data will have 15.
For example: Test data = data[0]-data[15], Training data = data[15]-data[150]
             Test data = data[15]-data[30], Training data = data[0]-data[15] + data[30]-data[150]
'''
print("Execution is starting!")
print("Learning Rate: ", learning_rate)
print("Max number of iteration: ", max_iter)
for i in range(10):
    # it takes number of 15 samples from index = i*15
    test_data = data[i*15:][:15]
    # first, it takes samples until index = i*15, then adds the rest of the data,e. g. after test_data
    training_data = data[:i*15] + data[(i+1)*15:]

    attributes, types = split(training_data)
    N = len(training_data)  # number of samples
    w = train(N, K, d, attributes, types)  # training the data, it returns weight vectors

    attributes, types = split(test_data)
    predicted = []
    for k in range(15):
        p0 = p1 = p2 = 0
        for j in range(4):
            p0 += w[0][j] * attributes[k][j]
            p1 += w[1][j] * attributes[k][j]
            p2 += w[2][j] * attributes[k][j]

        result = biggest(p0, p1, p2)
        predicted.append(result)
        confusion_matrix = confusion(confusion_matrix, result, types[k])

    print(i + 1, end='')
    print(".", end='')
    acc_ratio.append(accuracy(predicted, types))

print("Confusion Matrix: ")
print(confusion_matrix)
print("Accuracy rate for 10-fold cross validation:", "{0:.3f}%".format((sum(acc_ratio)/len(acc_ratio))*100))
