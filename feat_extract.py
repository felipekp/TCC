# --- general imports
import graphviz
import matplotlib as plt
import numpy as np
import os
import pandas as pd
import pydot
import pydotplus

from matplotlib import pyplot

from sklearn.decomposition import PCA

from sklearn import tree
from sklearn.externals.six import StringIO

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.feature_selection import RFE, SelectKBest, f_regression
from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier, plot_importance

from sklearn.preprocessing import MinMaxScaler

# np.set_printoptions(threshold=np.nan)
pd.set_option('display.max_rows', 200000)

# --- logging
import logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(funcName)20s() %(levelname)-8s %(message)s',
                    datefmt='%d-%m %H:%M:%S',
                    filename='logs/feat_extract.log',
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


def decision_tree(dataset, target_dataset, dataplot1):
    logger.info('Decision Tree Classifier')
    model = tree.DecisionTreeClassifier()
    model.fit(dataset, target_dataset)
    # ------ exporting the tree
    print '--------- Result: Decision Tree'
    print len(model.feature_importances_)
    # dot_data = StringIO()
    # tree.export_graphviz(model, out_file=dot_data)
    # graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    # graph.write_pdf("out/dec_tree.pdf")


def feature_importance(dataset, target_dataset, dataplot1):
    logger.info('Feature importance')
    # ------ feature extraction
    model = ExtraTreesClassifier()
    model.fit(dataset, target_dataset)

    # ------ printing results
    print '--------- Result: Feature Importance'
    print(model.feature_importances_)


def grad_boosting_classifier(dataset, target_dataset, dataplot1):
    logger.info('Gradient boosting classifier')
    # ------ creates and trains the classifier
    model = XGBClassifier()
    model.fit(dataset, target_dataset)

    # ------ feature importance
    print '--------- Result: Gradient boosting classifier'
    print(model.feature_importances_)

    # ------ plot
    # pyplot.bar(range(len(model.feature_importances_)),
    #            model.feature_importances_)
    plot_importance(model, grid=False)
    pyplot.show()


def pca(dataset, target_dataset, dataplot1):
    logger.info('PCA')
    # ------ feature extraction
    n_components = 25
    # TODO: create a while loop that evaluates the 'best' number of components by checking if sum_fitpca.explanined... is greater than 0.999?
    fit_pca = PCA(n_components=n_components, whiten=True).fit(dataset1)  # 7 principal components
    # ------ printing results
    dataset_pca = fit_pca.fit_transform(dataset)
    print '--------- Result: PCA'
    print("Variance preserved: %s") % sum(fit_pca.explained_variance_ratio_)
    # print(fit_pca.components_) # prints the eigen vectors, each pca is a vector
    # ------ saves resulting dataset to a file
    df = pd.DataFrame(dataset_pca)
    # df['excess_ozone'] = target_dataset
    df['readings_ozone'] = dataplot1
    write_new_csv(df, 'pca.csv')

    # ------ calculates cumulative variance
    # temp = []
    # temp.append(fit_pca.explained_variance_ratio_[0])
    # i = 0
    # for item in fit_pca.explained_variance_ratio_[1:]:
    #     temp.append(temp[i]+item)
    #     i += 1
    # pyplot.ylabel('% of cumulative variance')
    # pyplot.xlabel('principal components')
    # pyplot.plot(range(len(fit_pca.explained_variance_ratio_)),temp, 'r')
    # pyplot.show()

    # ----- plots the variance ratio
    pyplot.ylabel('% of variance')
    pyplot.xlabel('principal components')
    pyplot.bar(range(len(fit_pca.explained_variance_ratio_)),fit_pca.explained_variance_ratio_)
    pyplot.show()


def recursive_feature_elim(dataset, target_dataset, dataplot1):
    logger.info('Recursive Feature Elimination with Logistic Regression')
    # ------ feature extraction
    model = LogisticRegression()
    rfe = RFE(model, 27)  # selects 27 features
    fit = rfe.fit(dataset, target_dataset)

    # ------ printing results
    print '--------- Result: Recursive feature elimination'
    print("Selected Features: %s") % fit.support_
    print("Feature Ranking: %s") % fit.ranking_


def univariate_selection(dataset, target_dataset, dataplot1):
    test = SelectKBest(score_func=f_regression, k=25)
    fit = test.fit(dataset, target_dataset)

    # ------ scores
    np.set_printoptions(precision=3)
    print(fit.scores_)
    features = fit.transform(dataset)
    # summarize selected features
    # print(features[0:5,:])

def write_new_csv(df, filename):
    """
        Saves the dataframe inside a new file in a new path (a folder with 'clean-' as prefix)
        :param df: dataframe with the modified data
        :param filename: filename from file being read (file name will stay the same)
        :param county: county number
        :return:
    """
    global start_year, end_year
    logging.info('Saving file into new folder')
    newpath = 'out/'
    if not os.path.exists(newpath):
        os.makedirs(newpath)

    df.to_csv(os.path.join(newpath, filename))


extr_feat_algs = {
    0: decision_tree,
    1: feature_importance,
    2: grad_boosting_classifier,
    3: pca,
    4: recursive_feature_elim,
    5: univariate_selection,
}


def main():
    logging.info('Started MAIN')
    start_year = '2000'
    end_year = '2016'
    site = '0069'
    algs_to_use = [0, 2, 3]

    # my_file = open('merged/max-merged_' + start_year + '-' + end_year + '.csv')
    my_file = open('datasets/kuwait.csv')
    # my_file = open('brian_phd/brian_dataset.csv')
    df = pd.read_csv(my_file, skipfooter=1, engine='python')
    df = df.set_index(df.columns[0])
    df.index.rename('id', inplace=True)

    logger.info('DO:Feature extraction')

    # --------- remove columns that do not contain _0069
    # remove_other_site_cols(df, site)

    # pick column to predict
    target_col = 23   # choses 44201 as target column TODO: CHANGE IT FOR THE KUWAIT DATASET

    # ----- reshaping the data
    dataset1 = df.fillna(0).values
    # deletes target_column data
    dataset1 = np.delete(dataset1, target_col, axis=1)
    dataset1 = dataset1.astype('float32')
    dataplot1 = df[df.columns[23]]
    dataplot1 = dataplot1.values.reshape(-1, 1)  # reshapes data for minmax scaler

    a = list(df)

    for i in range (len(a)):
        print i, a[i]

    scalerX = MinMaxScaler(feature_range=(0, 1))
    scalerY = MinMaxScaler(feature_range=(0, 1))
    
    dataset = scalerX.fit_transform(dataset1)
    dataplot = scalerY.fit_transform(dataplot1)
    

    # ----- modifies the target column so when its above standard (0.07) its 1 and else 0
    target_dataset = np.where(df[df.columns[23]] >= 0.07, 1, 0).astype('int')

    # ----- creates and trains feature extraction methods
    for alg in algs_to_use:
        # TODO: try and except for items inside algs_to_use
        extr_feat_algs[alg](dataset, target_dataset, dataplot1)

    logger.info('DONE:Feature extraction')
    logging.info('Finished MAIN')


if __name__ == "__main__":
    main()
