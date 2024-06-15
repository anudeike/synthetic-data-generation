import numpy as np
import pandas as pd
import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

"""
For this data: create a model that can accurately predict the grade point average of a student given 
the following features: gender, race/ethnicity, parental level of education, study habits etc


"""
def calculate_summary_statistics(data):

    pass

def linear_regression(data, columnsToDrop, targetColumn, test_size = 0.2):
    # define the features and target
    X = data.drop()
    pass

def log(msg):
    print(f"=> {datetime.datetime.now()}: {msg}")

def main():


    realData = pd.read_csv('real data\spotify_track_dataset.csv')
    print(realData.head())

    print(realData.columns)

    # using linear regression as a baseline

    # where there are null values, remove the entire row
    realData.dropna(inplace = True)

    # encode the categorical varliables (one hot encoding)
    onehotdata = pd.get_dummies(realData, columns=["track_genre"], drop_first=True)
    log("one hot encoding done")

    # define the features and the target - in this case it is popularity
    X = onehotdata.drop(columns = ["popularity", "artists", "album_name", "track_name", "track_id"])
    y = onehotdata.popularity

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train a linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Print the results
    log(f"Mean squared error: {mse}\nR2 score: {r2} ==> Model score: {model.score(X_test, y_test)}")

    # write the evaluations to a file with the parameters of the model 
    evaluations = pd.DataFrame({'Mean squared error': [mse], 'R2 score': [r2], 'Model score': f"{np.round(model.score(X_test, y_test) * 100, 2)}%"})
    evaluations.to_csv('linreg_evaluations.csv', index=False)


    pass

main()