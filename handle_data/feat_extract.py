# --- general imports
from collections import OrderedDict

import graphviz
import matplotlib as plt
import numpy as np
import os
import pandas as pd
import pydot
import pydotplus
import utils.utils as utils

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


# --- logging - always cleans the log when importing and executing this file
import logging
utils.setup_logger('logger_feat_extract', r'logs/feat_extract.log')
logger = logging.getLogger('logger_feat_extract')

# --- measuring time
import time

# --- global variables
global start_year
global end_year


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


def decision_tree(dataset, default_target, target_dataset, timesteps, col_names, extracted_output_path):
    logger.info('Decision Tree Classifier')

    temp_target_dataset = pd.DataFrame(target_dataset)
    target_bin_dataset = np.where(temp_target_dataset.values >= 0.07, 1, 0).astype('int')
    target_bin_dataset = target_bin_dataset.reshape(-1, 1)

    model = tree.DecisionTreeClassifier()
    model.fit(dataset, target_bin_dataset)
    # ------ exporting the tree
    print '--------- Result: Decision Tree'
    # print model.feature_importances_

    score_and_cols = zip([float(x) for x in model.feature_importances_], col_names[:len(col_names)-1])
    ten_most_important_features = sorted(score_and_cols, key=lambda t: t[0], reverse=True)[:10]
    # df = pd.DataFrame(dataset.reshape(-1, len(col_names)), columns=col_names)
    df = pd.DataFrame(data=dataset, columns=col_names)

    new_df = pd.DataFrame()
    for item in ten_most_important_features:
        new_df[item[1]] = df.pop(item[1])

    print('The ten most important features are:')
    print(ten_most_important_features)

    new_df['target_t+' + timesteps] = target_dataset
    new_df['target_t+0'] = default_target
    new_df.to_csv(extracted_output_path + 'decTree_' + start_year + '-' + end_year + '.csv')

    # following code just creates the decision tree in a pdf file.
    # dot_data = StringIO()
    # tree.export_graphviz(model, out_file=dot_data)
    # graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    # graph.write_pdf("out/dec_tree.pdf")


# def feature_importance(dataset, target_dataset, dataplot1):
#     logger.info('Feature importance')
#     # ------ feature extraction
#     model = ExtraTreesClassifier()
#     model.fit(dataset, target_dataset)
#
#     # ------ printing results
#     print '--------- Result: Feature Importance'
#     print(model.feature_importances_)


# def grad_boosting_classifier(dataset, target_dataset, dataplot1):
#     logger.info('Gradient boosting classifier')
#     # ------ creates and trains the classifier
#     model = XGBClassifier()
#     model.fit(dataset, target_dataset)
#
#     # ------ feature importance
#     print '--------- Result: Gradient boosting classifier'
#     print(model.feature_importances_)
#
#     # ------ plot
#     # pyplot.bar(range(len(model.feature_importances_)),
#     #            model.feature_importances_)
#     plot_importance(model, grid=False)
#     pyplot.show()


def pca(dataset, default_target, target_dataset, timesteps, col_names, extracted_output_path):
    #change to dataset, target_bin_dataset, target_dataset
    logger.info('PCA')
    global start_year, end_year
    # ------ feature extraction
    n_components = 10
    fit_pca = PCA(n_components=n_components, whiten=True)
    dataset_pca = fit_pca.fit_transform(dataset)
    # ------ printing results
    print '--------- Result: PCA'
    print("Variance preserved: %s") % sum(fit_pca.explained_variance_ratio_)
    # print(fit_pca.components_) # prints the eigen vectors, each pca is a vector
    # ------ saves resulting dataset to a file
    df = pd.DataFrame(data= dataset_pca, columns= ['principal_comp_' + str(x) for x in range(10)])
    # df['excess_ozone?'] = target_bin_dataset

    df['target_t+' + timesteps] = target_dataset
    df['target_t+0'] = default_target

    df.to_csv(extracted_output_path + 'pca_' +start_year + '-' + end_year + '.csv')

    # ------ calculates cumulative variance
    temp = []
    temp.append(fit_pca.explained_variance_ratio_[0])
    i = 0
    for item in fit_pca.explained_variance_ratio_[1:]:
        temp.append(temp[i]+item)
        i += 1
    pyplot.ylabel('% of cumulative variance')
    pyplot.xlabel('principal components')
    pyplot.plot(range(len(fit_pca.explained_variance_ratio_)),temp, 'r')
    pyplot.show()

    # ----- plots the variance ratio
    pyplot.ylabel('% of variance')
    pyplot.xlabel('principal components')
    pyplot.bar(range(len(fit_pca.explained_variance_ratio_)),fit_pca.explained_variance_ratio_)
    pyplot.show()


# def recursive_feature_elim(dataset, target_dataset, dataplot1):
#     logger.info('Recursive Feature Elimination with Logistic Regression')
#     # ------ feature extraction
#     model = LogisticRegression()
#     rfe = RFE(model, 27)  # selects 27 features
#     fit = rfe.fit(dataset, target_dataset)
#
#     # ------ printing results
#     print '--------- Result: Recursive feature elimination'
#     print("Selected Features: %s") % fit.support_
#     print("Feature Ranking: %s") % fit.ranking_

#
# def univariate_selection(dataset, target_dataset, dataplot1):
#     test = SelectKBest(score_func=f_regression, k=25)
#     fit = test.fit(dataset, target_dataset)
#
#     # ------ scores
#     np.set_printoptions(precision=3)
#     print(fit.scores_)
#     features = fit.transform(dataset)
#     # summarize selected features
#     # print(features[0:5,:])
#
# def write_new_csv(df, filename):
#     """
#         Saves the dataframe inside a new file in a new path (a folder with 'clean-' as prefix)
#         :param df: dataframe with the modified data
#         :param filename: filename from file being read (file name will stay the same)
#         :param county: county number
#         :return:
#     """
#     global start_year, end_year
#     logging.info('Saving file into new folder')
#     newpath = 'out/'
#     if not os.path.exists(newpath):
#         os.makedirs(newpath)
#
#     df.to_csv(os.path.join(newpath, filename))


extr_feat_algs = {
    0: decision_tree,
    # 1: feature_importance,
    # 2: grad_boosting_classifier,
    3: pca,
    # 4: recursive_feature_elim,
    # 5: univariate_selection,
}


def extract_features(p_start_year, p_end_year, algs_to_use, county, extracted_input_path, extracted_output_path, predict_var, timesteps, state='48', site='0069'):
    logging.info('Started MAIN')
    global start_year, end_year
    start_year = p_start_year
    end_year = p_end_year

    filename = str(extracted_input_path + start_year + '-' + end_year + '.csv')
    # my_file = open('brian_phd/brian_dataset.csv')
    df = pd.read_csv(filename, engine='python')
    df = df.set_index(df.columns[0])
    df.index.rename('id', inplace=True)

    logger.info('DO:Feature extraction')

    # --------- remove columns that do not contain _0069
    # remove_other_site_cols(df, site)

    # pick column to predict
    target_col = predict_var + '_t+' + timesteps   # selects last column as target
    default_col = predict_var + '_t+0'
    target_dataset = pd.DataFrame(df.pop(target_col))
    default_target = pd.DataFrame(df.pop(default_col))
    # ----- modifies the target column so when its above standard (0.07) its 1 and else 0. This is actually only used inside Decision Tree for now.

    print 'Total parameters/columns:', len(df.columns)-1
    print 'The target parameter is:', target_col
    print 'The default column is (target_col not shifted):', default_col

    # ----- reshaping the data
    dataset = df.fillna(0).values
    dataset = dataset.astype('float32')
    target_dataset = target_dataset.values.reshape(-1, 1)  # reshapes data for minmax scaler

    # MUST re-scale for PCA and Dec tree
    scalerX = MinMaxScaler(feature_range=(0, 1))

    dataset = scalerX.fit_transform(dataset)

    # ----- creates and trains feature extraction methods
    for alg in algs_to_use:
        extr_feat_algs[alg](dataset, default_target, target_dataset, timesteps, df.columns, extracted_output_path)

    logger.info('DONE:Feature extraction')
    logging.info('Finished MAIN')


