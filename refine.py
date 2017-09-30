# --- general imports
import datetime as DT
import matplotlib as plt
import numpy as np
import os
import pandas as pd

pd.set_option('display.max_rows', 200000)

# --- logging
import logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(funcName)20s() %(levelname)-8s %(message)s',
                    datefmt='%d-%m %H:%M:%S',
                    filename='refine.log',
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

# --- global variables
global start_year
global end_year

# --- START Functions


def handle_outliers(df):
    """
        There are no silver bullets for this issue...
        Removes negative values from readings that should not have negative values
    """
    logging.warning('Removing outliers (negative values and putting 0)')

    df = df.clip_lower(0)

    return df


def calc_dispersion_missing(df):
    """
        Calculates dispersion of missing data and interpolates for small and medium gaps (small 0 to 30 days, medium 30 to 365 days, large 365+ days). Also sets True to a flag if it was not possible to fill missing values with small and medium methods (large gaps not being handled).
        :param df: pandas dataframe with date and other values from parameters
        :return: tuple with: modified dataframe or not and a flag saying if the given column/dataframe should be deleted or not (True deletes the column because there is not enough data for interpolate and False means the column contains data for all days)
    """
    new_df = df.isnull().astype(int).groupby(
        df.notnull().astype(int).cumsum()).sum()

    # maximum consecutive days without readings
    if new_df.max().astype(int) > 365:
        # print new_df.max()
        logger.critical(
            'LARGE. Cannot interpolate because max size is over 365')
        return df, True

    logger.warning(
        'SMALL. Maximum size is under one year. So, we can interpolate some parts using a 30 days limit for linear')
    df = df.interpolate(limit_direction='both', limit=30)

    logger.warning(
        'MEDIUM. Using mean from previous and next year to fill this gap')
    # size of window to look for values (2 means 2 years ahead and 2 years
    # behind)
    years = 2
    # for year gap (takes values from next and previous year) (if it was 30,
    # would use values from previous and next month)
    gap = 365
    df[df.isnull()] = np.nanmean(
        [df.shift(x).values for x in range(-gap * years, gap * (years + 1), gap)], axis=0)

    new_df = df.isnull().astype(int).groupby(
        df.notnull().astype(int).cumsum()).sum()
    if new_df.max().astype(int) > 0:  # means that there is not enough data from previous or next year to help fit this point, thus there is still nan values in the dataframe (will return with True because this column will be dropped!)
        logger.critical(
            'Cannot interpolate with medium nor small methods because there is not enough data from previous or next years!')
        return df, True
    # print new_df.max()
    # print df

    return df, False


def workaround_interpolate(df):
    """
        Loops over all columns and substitutes or deletes it based if the calc_dispersion_missing function was able to interpolate the gaps
        Based on this: https://stackoverflow.com/questions/32850185/change-value-if-consecutive-number-of-certain-condition-is-achieved-in-pandas
        :param df: pandas dataframe with date and other values from parameters
        :return: modified dataframe with all columns that were interpolated (only complete data gets returned here)
    """
    for col in df.columns:
        df[col], delete_col = calc_dispersion_missing(df[col])
        if delete_col:
            del df[col]
    return df


def handle_interpolate(df):
    """
        Calls a workaround function
        :return: modified dataframe from workaround_interpolate
    """
    logger.info('Handling interpolate')

    return workaround_interpolate(df)


def write_new_csv(df, filename, county):
    """
        Saves the dataframe inside a new file in a new path (a folder with 'clean-' as prefix)
        :param df: dataframe with the modified data
        :param filename: filename from file being read (file name will stay the same)
        :param county: county number
        :return:
    """
    global start_year, end_year
    logging.info('Saving file into new folder')
    newpath = '48/' + county + '/mean-refine-' + start_year + '-' + end_year
    if not os.path.exists(newpath):
        os.makedirs(newpath)

    df.to_csv(os.path.join(newpath, filename))


@timeit
def main():
    logging.info('Started MAIN')
    # temp_parameter = '43205'
    # to select folder:
    global start_year, end_year
    start_year = '2000'
    end_year = '2016'
    county = '113'
    root_dir = str('48/' + county + '/mean-clean-' +
                   start_year + '-' + end_year + '/')
    # parameters that will not go throught outlier removal
    not_remove_outlier = ['68105']

    # iterate over files inside folder
    for filename in os.listdir(root_dir):
        complete_path = os.path.join(root_dir, filename)

        df = pd.read_csv(complete_path, skipfooter=1, engine='python')
        df = df.set_index(['date'])

        if df.empty:
            logger.warning('DataFrame with file: %s is empty. Passing', complete_path)
            continue

        parameter = filename.split('.')[0]  # sets the parameter

        # if parameter != temp_parameter: #### TEMPORARY FOR TESTING
        # 	continue ### TEMPORARY FOR TESTING

        logger.info('DO:DataFrame in file: %s will be modified', complete_path)

        # --------- handling outliers:
        if parameter not in not_remove_outlier:
            df = handle_outliers(df)
        # --------- handling interpolate:
        df = handle_interpolate(df)
        # --------- writes file to new folder with prefix: 'refine-'
        write_new_csv(df, filename, county)

        logger.info('DONE:DataFrame in file: %s was modified', complete_path)

    logging.info('Finished MAIN')


if __name__ == "__main__":
    main()
