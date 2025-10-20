import pandas as pd

# Daten laden
data = pd.read_csv('../data/train.csv')
data = data.dropna()

data=data.drop(columns=['t'])

for feature1 in data.columns:
    feature_x1 = data[feature1]
    feature1=feature1
    for feature2 in data.columns:
        feature2=feature2
        feature_x2 = data[feature2]
        if feature1 != feature2:
            if (feature_x1.corr(feature_x2) > 0.6):
                print(feature1, feature2, feature_x1.corr(feature_x2))