import numpy as np
import pandas as pd

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

def main():


    realData = pd.read_csv('real data\spotify_track_dataset.csv')
    print(realData.head())

    print(realData.columns)

    # using linear regression as a baseline

    # where there are null values, remove the entire row
    realData.dropna(inplace = True)

    # encode the categorical varliables (one hot encoding)
    onehotdata = pd.get_dummies(realData, columns=["album_name", "artists", "track_name", "track_genre"], drop_first=True)
    print(onehotdata.head())
    pass

main()