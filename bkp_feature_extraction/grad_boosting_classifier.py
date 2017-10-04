'''
code modified from: https://machinelearningmastery.com/feature-importance-and-feature-selection-with-xgboost-in-python/
'''
from numpy import loadtxt
import numpy as np
from xgboost import XGBClassifier
from matplotlib import pyplot
import pandas as pd


def remove_other_site_cols(df, site):
    for col in df.columns:
        # print col.split('_')[1]
        if col.split('_')[1] != site:
            del df[col]


def main():
    start_year = '2000'
    end_year = '2016'
    site = '0069'

    my_file = open('merged/max-merged_' + start_year + '-' + end_year + '.csv')
    df = pd.read_csv(my_file, skipfooter=1, engine='python')
    df = df.set_index(df.columns[0])
    df.index.rename('id', inplace=True)

    # --------- remove columns that do not contain _0069
    remove_other_site_cols(df, site)
    # --------- preprocess data
    a = list(df)

    for i in range(len(a)):
        print i, a[i]

    # pick column to predict
    try:
        target_col = int(
            raw_input("Select the column number to predict (default = 23): "))
    except ValueError:
        target_col = 23   # choses 44201 as target column

    # ------ reshaping the data
    dataset1 = df.fillna(0).values
    dataplot1 = dataset1[0:, target_col]  # extracts the target_col
    dataplot1 = dataplot1.reshape(-1, 1)  # reshapes data
    # deletes target_column data
    dataset1 = np.delete(dataset1, target_col, axis=1)
    dataset1 = dataset1.astype('float32')

    # ------ modifies the target column so when its above standard (0.07) its 1 and else 0
    newdataset = np.where(df[df.columns[23]] >= 0.07, 1, 0).astype('int')

    # ------ creates and trains the classifier
    model = XGBClassifier()
    model.fit(dataset1, newdataset)

    # ------ feature importance
    print(model.feature_importances_)

    # ------ plot
    pyplot.bar(range(len(model.feature_importances_)), model.feature_importances_)
    pyplot.show()


if __name__ == "__main__":
    main()
