from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import numpy as np
import pandas as pd
import csv
import random
import math
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris


def loadDataset(filename, split):
    data = []
    with open(filename, 'r') as csvfile:
        lines = csvfile.readlines()
        for x in lines:
            if x.strip() == "":
                continue
            r = x.strip().split(",")
            for y in range(len(r)-1):
                if is_number(r[y]):
                    r[y] = float(r[y])
            else:
                r[-1] = str(r[-1])
                data.append(r)
    size = len(data) * split
    trainingSet = data[:int(size)]
    testSet = data[int(size):]
    return trainingSet, testSet


def is_number(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def KNN(trainingSet, testSet, k):
    predictions = []
    for x in testSet:
        output = []
        distances = []
        neighbors = []
        for y in trainingSet:
            dist = 0  # row1, row2
            distance = 0.0
            for i in range(len(x)-1):
                distance += (x[i] - y[i])**2
            dist = math.sqrt(distance)
            distances.append((y, dist))
        distances.sort(key=lambda x: x[1])
        for i in range(k):
            neighbors.append(distances[i][0])
        output = [row[-1] for row in neighbors]
        pred = max(set(output), key=output.count)
        predictions.append(pred)

    return predictions
