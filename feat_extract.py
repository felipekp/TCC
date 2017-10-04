# --- general imports
import graphviz
import matplotlib as plt
import numpy as np
import pandas as pd
import pydot
import pydotplus
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


def decision_tree(dataset1, target_dataset):
    logger.info('Decision Tree Classifier')
    model = tree.DecisionTreeClassifier()
    model.fit(dataset1, target_dataset)
    # ------ exporting the tree
    dot_data = StringIO()
    tree.export_graphviz(model, out_file=dot_data)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf("out/dec_tree.pdf")

def feature_importance(dataset1, target_dataset):
    logger.info('Feature importance')
    # ------ feature extraction
    model = ExtraTreesClassifier()
    model.fit(dataset1, target_dataset)
    
    # ------ printing results
    print(model.feature_importances_)

def grad_boosting_classifier(dataset1, target_dataset):
    logger.info('Gradient boosting classifier')
    # ------ creates and trains the classifier
    model = XGBClassifier()
    model.fit(dataset1, target_dataset)

    # ------ feature importance
    print(model.feature_importances_)

    # ------ plot
    pyplot.bar(range(len(model.feature_importances_)), model.feature_importances_)
    pyplot.show()

def pca(dataset1, target_dataset):
    logger.info('PCA')
    # ------ feature extraction
    pca = PCA(n_components=25) # 25 principal components
    fit = pca.fit(dataset1)
    # ------ printing results
    print("Variance: %s") % fit.explained_variance_ratio_
    print(fit.components_)

def recursive_feature_elim(dataset1, target_dataset):
    logger.info('Recursive feature elimination')
    # ------ feature extraction
    model = LogisticRegression()
    rfe = RFE(model, 25) # selects 25 features
    fit = rfe.fit(dataset1, target_dataset)

    # ------ printing results
    print("Selected Features: %s") % fit.support_
    print("Feature Ranking: %s") % fit.ranking_

extr_feat_algs = {   
        0 : decision_tree,
        1 : feature_importance,
        2 : grad_boosting_classifier,
        3 : pca,
        4 : recursive_feature_elim,
    }

def main():
    logging.info('Started MAIN')
    start_year = '2000'
    end_year = '2016'
    site = '0069'
    algs_to_use = [0, 1, 2, 3, 4]

    my_file = open('merged/max-merged_' + start_year + '-' + end_year + '.csv')
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
    dataset1 = np.delete(dataset1, target_col, axis=1) # deletes target_column data
    dataset1 = dataset1.astype('float32')
    
    # ----- modifies the target column so when its above standard (0.07) its 1 and else 0
    target_dataset = np.where(df[df.columns[23]] >= 0.07, 1, 0).astype('int')
    
    # ----- creates and trains feature extraction methods
    for alg in algs_to_use:
        # TODO: try and except for items inside algs_to_use
        extr_feat_algs[alg](dataset1, target_dataset)

    logger.info('DONE:Feature extraction')
    logging.info('Finished MAIN')


if __name__ == "__main__":
    main()
