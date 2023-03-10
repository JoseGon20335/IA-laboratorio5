# LABORATORIO 5
# JOSE MIGUEL GONZALEZ Y GONZALEZ - 20335
# DIEGO PERDOMO - 20

# TASK 1.1
# K-NEAREST NEIGHBORS
# Importing libraries
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
from KNearest import loadDataset

# Importing dataset
print("TASK 1.1 SIN LIBRERIAS")
print("Importing dataset")

file = 'dataset_phishing.csv'
trainingSet = []
testSet = []
trainingSet, testSet = loadDataset(file, 0.8, trainingSet, testSet)
print('Train set: ' + len(trainingSet))
print('Test set: ' + len(testSet))
print("")
