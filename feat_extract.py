# --- general imports
import graphviz
import matplotlib as plt
import numpy as np
import pandas as pd
import pydot
import pydotplus
import os
from matplotlib import pyplot

from sklearn.decomposition import PCA

from sklearn import tree
from sklearn.externals.six import StringIO

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier

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


def decision_tree(dataset1, target_dataset, data_plot):
    logger.info('Decision Tree Classifier')
    model = tree.DecisionTreeClassifier()
    model.fit(dataset1, target_dataset)
    # ------ exporting the tree
    print '--------- Result: Decision Tree'
    dot_data = StringIO()
    tree.export_graphviz(model, out_file=dot_data)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf("out/dec_tree.pdf")


def feature_importance(dataset1, target_dataset, data_plot):
    logger.info('Feature importance')
    # ------ feature extraction
    model = ExtraTreesClassifier()
    model.fit(dataset1, target_dataset)

    # ------ printing results
    print '--------- Result: Feature Importance'
    print(model.feature_importances_)


def grad_boosting_classifier(dataset1, target_dataset, data_plot):
    logger.info('Gradient boosting classifier')
    # ------ creates and trains the classifier
    model = XGBClassifier()
    model.fit(dataset1, target_dataset)

    # ------ feature importance
    print '--------- Result: Gradient boosting classifier'
    print(model.feature_importances_)

    # ------ plot
    pyplot.bar(range(len(model.feature_importances_)),
               model.feature_importances_)
    pyplot.show()


def pca(dataset1, target_dataset, data_plot):
    logger.info('PCA')
    # ------ feature extraction
    # TODO: create a while loop that evaluates the best number of components by checking if sum_fitpca.explanined... is greater than 0.999?
    fit_pca = PCA(n_components=7, whiten=True).fit(dataset1)  # 7 principal components
    # ------ printing results
    dataset_pca = fit_pca.transform(dataset1)
    print '--------- Result: PCA'
    print("Variance preserved: %s") % sum(fit_pca.explained_variance_ratio_)
    # print(fit_pca.components_) # prints the eigen vectors, each pca is a vector
    # ------ saves resulting dataset to a file
    df = pd.DataFrame(dataset_pca)
    # df['excess_ozone'] = target_dataset
    df['readings_ozone'] = data_plot
    write_new_csv(df, 'pca.csv')


def recursive_feature_elim(dataset1, target_dataset, data_plot):
    logger.info('Recursive feature elimination')
    # ------ feature extraction
    model = LogisticRegression()
    rfe = RFE(model, 25)  # selects 25 features
    fit = rfe.fit(dataset1, target_dataset)

    # ------ printing results
    print '--------- Result: Recursive feature elimination'
    print("Selected Features: %s") % fit.support_
    print("Feature Ranking: %s") % fit.ranking_


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
}


def main():
    logging.info('Started MAIN')
    start_year = '2000'
    end_year = '2016'
    site = '0069'
    algs_to_use = [3]

    my_file = open('merged/max-merged_' + start_year + '-' + end_year + '.csv')
    # my_file = open('brian_phd/brian_dataset.csv')
    df = pd.read_csv(my_file, skipfooter=1, engine='python')
    df = df.set_index(df.columns[0])
    df.index.rename('id', inplace=True)

    logger.info('DO:Feature extraction')

    # --------- remove columns that do not contain _0069
    remove_other_site_cols(df, site)

    # pick column to predict
    target_col = 23   # choses 44201 as target column

    # ----- reshaping the data
    dataset1 = df.fillna(0).values
    # deletes target_column data
    dataset1 = np.delete(dataset1, target_col, axis=1)
    dataset1 = dataset1.astype('float32')
    data_plot = df[df.columns[23]]

    # ----- modifies the target column so when its above standard (0.07) its 1 and else 0
    target_dataset = np.where(df[df.columns[23]] >= 0.07, 1, 0).astype('int')

    # ----- creates and trains feature extraction methods
    for alg in algs_to_use:
        # TODO: try and except for items inside algs_to_use
        extr_feat_algs[alg](dataset1, target_dataset, data_plot)

    logger.info('DONE:Feature extraction')
    logging.info('Finished MAIN')


if __name__ == "__main__":
    main()
