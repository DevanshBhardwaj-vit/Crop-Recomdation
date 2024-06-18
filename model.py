import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('Crop_Recommendation.csv')

df.rename(columns={'Nitrogen': 'N', 'Phosphorus': 'P', 'Potassium': 'K'}, inplace=True)

features = df[['N', 'P', 'K', 'Temperature', 'Humidity', 'pH_Value', 'Rainfall']]
label = df['Crop']

from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn import tree

from sklearn.model_selection import train_test_split

Xtrain, Xtest, Ytrain, Ytest = train_test_split(features, label, test_size=0.2, random_state=2)

from sklearn.ensemble import RandomForestClassifier

RF = RandomForestClassifier(n_estimators=20, random_state=0)
RF.fit(Xtrain, Ytrain)

predicted_values = RF.predict(Xtest)

import pickle

pickle.dump(RF, open("model.pkl", "wb"))


def recommend_crops(soil_features):
    """
    Given soil features, recommend suitable crops.

    Parameters:
    soil_features (list): A list of soil features [feature1, feature2, ...]

    Returns:
    list: A list of recommended crops
    """
    # Convert input to the format expected by the model
    input_features = np.array(soil_features).reshape(1, -1)

    # Predict using the loaded model
    recommendations = RF.predict(input_features)

    return recommendations
