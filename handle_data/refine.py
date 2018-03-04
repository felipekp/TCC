"""
    This file constains methods to interpolate and handle with some outliers.
"""
# --- general imports
import datetime as DT
import matplotlib as plt
import numpy as np
import os
import pandas as pd
import utils.utils as utils

pd.set_option('display.max_rows', 200000)

# --- logging - always cleans the log when importing and executing this file
import logging
utils.setup_logger('logger_refine', r'logs/refine.log')
logger = logging.getLogger('logger_refine')

# --- global variables
global start_year
global end_year

# --- START Functions


def handle_outliers(df):
    """
        There are no silver bullets for this issue... TODO: maybe outliers should be handled before calculating the averages*
        Removes negative values from readings that should not have negative values
    """
    logger.warning('Removing outliers (negative values and putting 0)')

    df = df.clip_lower(0)

    return df


def calc_dispersion_missing(df):
    """
        Calculates dispersion of missing data and interpolates for small and medium gaps (small 0 to 10 gaps, medium 10 to 365 gaps, large 365+ days). Also sets True to a flag if it was not possible to fill missing values with small and medium methods.
        :param df: pandas dataframe with date and other values from parameters
        :return: tuple with: modified dataframe or not and a flag saying if the given column/dataframe should be deleted or not (True deletes the column because there is not enough data for interpolate and False means the column contains data for all days)
    """
    new_df = df.isnull().astype(int).groupby(
        df.notnull().astype(int).cumsum()).sum()

    # maximum consecutive days without readings
    if new_df.max().astype(int) > 365:
        # print new_df.max()
        logger.critical(
            'LARGE. Cannot interpolate because max size is over 365 (bigger gap is over 365 days)')
        return df, True

    logger.warning(
        'SMALL. Maximum size is under one year. So, we can interpolate some parts using a 10 gaps (days or 8h average, depends on the dataset) limit for linear method')
    df = df.interpolate(limit_direction='both', limit=10)

    logger.warning(
        'MEDIUM. Using mean from previous and next year to fill this gap, if possible')
    # size of window to look for values (2 means 2 years ahead and 2 years
    # behind)
    years = 7
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
    logger.info('Workaround interpolate')
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


@utils.timeit
def refine_data(p_start_year, p_end_year, county, refine_input_path, refine_output_path, state='48', site='0069'):
    """
    """
    logging.info('Started MAIN')
    # temp_parameter = '43205'
    # to select folder:
    global start_year, end_year
    start_year = p_start_year
    end_year = p_end_year
    root_dir = str(state + '/' + county+ '/' + refine_input_path + start_year + '-' + end_year + '/')
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

        # --------- writes file to new folder
        if len(df.columns) >= 1:
            utils.write_new_csv(df, refine_output_path, filename, county, state, start_year, end_year)
        else:
            logger.warning('Nothing to write for parameter: %s, not possible to interpolate (the file would be empty)', parameter)

        logger.info('DONE:DataFrame in file: %s was modified', complete_path)

    logging.info('Finished MAIN')

#
# if __name__ == "__main__":
#     main()
