import numpy as np
import pandas as pd
import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
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

def preprocess_data(data, categtorical_variables, columnsToDrop, targetColumn):
    
    # encode the categorical variables
    onehotdata = pd.get_dummies(data, columns=categtorical_variables, drop_first=True)
    log("one hot encoding done")

    # define the features and the target - in this case it is popularity
    X = onehotdata.drop(columns = columnsToDrop)
    y = onehotdata[targetColumn]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

# todo: finish the function --> add accurate logs
def linear_regression(data):
    
    # preprocess the data
    X_train, X_test, y_train, y_test = preprocess_data(data, ["track_genre"], ["popularity", "artists", "album_name", "track_name", "track_id"], "popularity")

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
    pass

def load_data(path = "real data\spotify_track_dataset.csv"):
    log(f"Loading Data from: {path}")
    
    return pd.read_csv(path)


def main():

    # load the data
    realData = load_data()

    log("Data loaded")

    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = preprocess_data(realData, ["track_genre"], ["popularity", "artists", "album_name", "track_name", "track_id"], "popularity")

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


    # do the same with logistic regression
    logRegModel = LogisticRegression()
    logRegModel.fit(X_train, y_train)

    logRegPredictions = logRegModel.predict(X_test)

    # error -- need to pass and index
    log("Logistic Regression Evaluation")
    log(f"Mean squared error: {mean_squared_error(y_test, logRegPredictions)}\nR2 score: {r2_score(y_test, logRegPredictions)} ==> Model score: {logRegModel.score(X_test, y_test)}")

    # write the evaluations to a file with the parameters of the model
    evaluations = pd.DataFrame({'Mean squared error': [mean_squared_error(y_test, logRegPredictions)], 'R2 score': [r2_score(y_test, logRegPredictions)], 'Model score': f"{np.round(logRegModel.score(X_test, y_test) * 100, 2)}%"})
    evaluations.to_csv('logreg_evaluations.csv', index=False)


    pass

main()