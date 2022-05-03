# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression


data = pd.read_csv('iris.csv')


variety_mappings = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}


data = data.replace(['Setosa', 'Versicolor' , 'Virginica'],[0, 1, 2])

X = data.iloc[:, 0:-1] # Extracting the independent variables
y = data.iloc[:, -1] # Extracting the target/dependent variable

logreg = LogisticRegression() #
logreg.fit(X, y)


def classify(a, b, c, d):
    arr = np.array([a, b, c, d]) # Convert to numpy array
    arr = arr.astype(np.float64) # Change the data type to float
    query = arr.reshape(1, -1) # Reshape the array
    prediction = variety_mappings[logreg.predict(query)[0]] # Retrieve from dictionary
    return prediction # Return the prediction
