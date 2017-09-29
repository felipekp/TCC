# --- general imports
import datetime as DT
import graphviz
import matplotlib as plt
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pydot
import pydotplus

from IPython.display import Image
from sklearn import preprocessing
from sklearn import tree
from sklearn import utils
from sklearn.externals.six import StringIO
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from xgboost import plot_tree


# np.set_printoptions(threshold=np.nan)
pd.set_option('display.max_rows', 200000)

# --- logging
import logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(funcName)20s() %(levelname)-8s %(message)s',
                    datefmt='%d-%m %H:%M:%S',
                    filename='decision_tree.log',
                    filemode='w')
logger = logging.getLogger(__name__)

# --- measuring time
import time


def timeit(method):

    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        print '%r %2.2f sec' % \
              (method.__name__, te - ts)
        return result

    return timed


def remove_other_site_cols(df, site):
    for col in df.columns:
        # print col.split('_')[1]
        if col.split('_')[1] != site:
            del df[col]

    # print df.head()


@timeit
def main():
    logging.info('Started MAIN')
    start_year = '2000'
    end_year = '2016'
    site = '0069'

    my_file = open('merged/merged_' + start_year + '-' + end_year + '.csv')
    df = pd.read_csv(my_file, skipfooter=1, engine='python')
    df = df.set_index(df.columns[0])
    df.index.rename('id', inplace=True)

    logger.info('DO:Decision Tree Code')
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

    dataset1 = df.fillna(0).values
    dataplot1 = dataset1[0:, target_col]  # extracts the target_col
    dataplot1 = dataplot1.reshape(-1, 1)  # reshapes data
    # deletes target_column data
    dataset1 = np.delete(dataset1, target_col, axis=1)
    dataset1 = dataset1.astype('float32')

    newdataset = np.where(df[df.columns[23]] >= 0.07, 1, 0).astype('int')

    # print '---------------'
    # print newdataset
    # print '-------'
    # print dataset1

    # exit()

    # scalerX = MinMaxScaler(feature_range=(0, 1))
    # scalerY = MinMaxScaler(feature_range=(0, 1))
    # dataset = scalerX.fit_transform(dataset1)
    # dataplot = scalerY.fit_transform(dataplot1)

    # dataplot = dataplot[:len(dataset)]
    # dataset = dataset[:len(dataplot)]

    # print len(dataplot)
    # print len(dataset)

    # for item in X:
    #   print item

    # print len(dataset1)

    # exit()

    # print dataplot1
    # exit()
    lab_enc = preprocessing.LabelEncoder()
    # encoded_dataset1 = lab_enc.fit_transform(dataset1)
    # encoded_dataplot1 = lab_enc.fit_transform(dataplot1.ravel())
    # encoded_dataplot1 = lab_enc.fit_transform(newdataset)
    # print(utils.multiclass.type_of_target(encoded_dataplot1))

    model = tree.DecisionTreeClassifier()
    model.fit(dataset1, newdataset)

    dot_data = StringIO()
    tree.export_graphviz(model, out_file=dot_data)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf("newnew.pdf")
    # --------- create tree

    logger.info('DONE:Decision Tree Code')

    logging.info('Finished MAIN')


if __name__ == "__main__":
    main()
