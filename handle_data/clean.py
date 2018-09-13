# --- general imports
import datetime as DT
import matplotlib as plt
import numpy as np
import os
import pandas as pd
import utils.utils as utils

# TODO: functions with calc_daily mean and stuff.. are all almost the same.. generalize.
# TODO: need to remove outliers before calculating mean.

pd.set_option('display.max_rows', 200000) # so pandas prints more rows

# --- logging - always cleans the log when importing and executing this file
import logging
utils.setup_logger('logger_clean', r'logs/clean.log')
logger = logging.getLogger('logger_clean')

# --- global variables
# global variables were used because of .apply() function used inside calc_stats, otherwise I wouldve used these two as normal parameters. They receive value inside 'main()'
global start_year
global end_year

# --- START Functions
def clean_datetime_site_daily(df):
    """
        Clears data in datetime and site field of a given pandas dataframe.
        site becomes: 'site' containing only the site code
        datetime becomes: 'date' containing only year-month-day
        value is repeated.
        :param df: a pandas dataframe with columns: datetime, site and value
        :return: a modified dataframe with columns: date, site and value
    """
    logger.info('Creating new DataFrame with site, date and repeating the value')

    new_df = df.groupby(['site', 'datetime'])
    date = list()
    sitenum = list()

    for leftf, rightf in new_df:
        sitenum.append(str(leftf[0])[8:])
        date.append(pd.Timestamp(DT.datetime(int(str(leftf[1])[0:4]), int(str(leftf[1])[4:6]), int(str(leftf[1])[6:8]))))

    temp_df = pd.DataFrame()
    temp_df['site'] = sitenum
    temp_df['date'] = date
    temp_df['value'] = df.value

    # return pd.DataFrame({'site': sitenum, 'date': date, 'value': df.value}) cant use this shorter version anymore because some issues with dictionary size
    return temp_df

def clean_datetime_site_8h(df):
    """
        Clears data in datetime and site field of a given pandas dataframe.
        site becomes: 'site' containing only the site code
        datetime becomes: 'date' containing only year-month-day hour:min:sec where hour is 00, 08 or 16.
        value is repeated.
        :param df: a pandas dataframe with columns: datetime, site and value
        :return: a modified dataframe with columns: date, site and value
    """
    logger.info('Creating new DataFrame with site, date and repeating the value')

    new_df = df.groupby(['site', 'datetime'])
    date = list()
    sitenum = list()

    for leftf, rightf in new_df:
        datetime_hour = int(str(leftf[1])[9:13])

        if datetime_hour <= 800:
            date.append(pd.Timestamp(DT.datetime(int(str(leftf[1])[0:4]), int(str(leftf[1])[4:6]), int(str(leftf[1])[6:8]), 0)))
        elif datetime_hour > 800 and datetime_hour <= 1600:
            date.append(pd.Timestamp(DT.datetime(int(str(leftf[1])[0:4]), int(str(leftf[1])[4:6]), int(str(leftf[1])[6:8]), 8)))
        elif datetime_hour > 1600 and datetime_hour <= 2300:
            date.append(pd.Timestamp(DT.datetime(int(str(leftf[1])[0:4]), int(str(leftf[1])[4:6]), int(str(leftf[1])[6:8]), 16)))
        else:
            logging.warning('Not supposed to happen--- bad things will happen since the size of date will not be the same as index. Must make sure each timestamp falls in one of the 3 categories in the code. Its probably a problem with the data given (the program is not being able to separate each datetime correctly in the categories)')

        sitenum.append(str(leftf[0])[8:])

    temp_df = pd.DataFrame()
    temp_df['site'] = sitenum
    temp_df['date'] = date
    temp_df['value'] = df.value

    # return pd.DataFrame({'site': sitenum, 'date': date, 'value': df.value}) cant use this shorter version anymore because some issues with dictionary size
    return temp_df


def calc_daily_mean(df):
    """
        Calculates daily mean and reindexes the data by (multiple indexes) date and sitenum (site was renamed to sitenum here)
        :param df: a pandas dataframe with columns: datetime, site and value
        :return: pandas dataframe reindexed with date and sitenum and daily mean in the value column
    """
    logger.info('Calculating daily MEAN and index by date and site')

    new_df = df.groupby(['site', 'date'])
    calc_mean = list()
    date = list()
    sitenum = list()

    for leftf, rightf in new_df:
        calc_mean.append(rightf['value'].mean())
        sitenum.append(leftf[0])
        date.append(leftf[1])

    # multi-indexing necessary for unstacking later
    return pd.DataFrame({'dailyVal': calc_mean}, index=[date, sitenum])

def calc_daily_max(df):
    """
        MAX USED ONLY FOR TESTING AND OZONE 44021.
        Calculates daily max and indexes the data by (multiple indexes) date and sitenum (site was renamed to sitenum here)
        :param df: a pandas dataframe with columns: datetime, site and value
        :return: pandas dataframe reindexed with date and sitenum and daily max in the value column
    """
    logger.info('Calculating daily MAX and index by date and sitenum')

    new_df = df.groupby(['site', 'date'])
    calc_max = list()
    date = list()
    sitenum = list()

    for leftf, rightf in new_df:
        calc_max.append(rightf['value'].max())
        sitenum.append(leftf[0])
        date.append(leftf[1])

    # multi-indexing necessary for unstacking later
    return pd.DataFrame({'dailyVal': calc_max}, index=[date, sitenum])


def calc_8h_average(df):
    """
        8h Average
        Calculates 8h average and indexes the data by (multiple indexes) date and sitenum (site was renamed to sitenum here)
        :param df: a pandas dataframe with columns: datetime, site and value
        :return: pandas dataframe reindexed with date and sitenum and daily max in the value column
    """
    logger.info('Calculating 8h average and index by date and sitenum')

    new_df = df.groupby(['site', 'date'])
    calc_8haverage = list()
    date = list()
    sitenum = list()


    for leftf, rightf in new_df:
        calc_8haverage.append(rightf['value'].mean())
        sitenum.append(leftf[0])
        date.append(leftf[1])

    # multi-indexing necessary for unstacking later
    return pd.DataFrame({'value': calc_8haverage}, index=[date, sitenum])


def calc_8h_maximum(df):
    """
        8h Average
        Calculates 8h average and indexes the data by (multiple indexes) date and sitenum (site was renamed to sitenum here)
        :param df: a pandas dataframe with columns: datetime, site and value
        :return: pandas dataframe reindexed with date and sitenum and daily max in the value column
    """
    logger.info('Calculating 8h average and index by date and sitenum')

    new_df = df.groupby(['site', 'date'])
    calc_8maximum = list()
    date = list()
    sitenum = list()


    for leftf, rightf in new_df:
        calc_8maximum.append(rightf['value'].max())
        sitenum.append(leftf[0])
        date.append(leftf[1])

    # multi-indexing necessary for unstacking later
    return pd.DataFrame({'value': calc_8maximum}, index=[date, sitenum])

def reindex_by_8h(df):
    """
        Assumes data is already sorted in the right sequency.
        Reindexes the data by 8h values. So, if one value was missing, now it will be added and have nan value
        :param df: a pandas dataframe with columns: datetime, site and value
        :return: a pandas dataframe with columns: datetime, site and value. However, it reindex the data by date and fills missing data (because the date didnt exist) for other columns with NaN value
    """
    global start_year, end_year
    logger.info('Reindexing by 8 hour daily (00 - 08 - 16 for each day)')

    # print pd.to_datetime(pd.date_range(start_year + '-01-01', end_year + '-12-31', freq='8H'))

    # exit()

    dates = pd.to_datetime(pd.date_range(start_year + '-01-01', end_year + '-12-31', freq='8H'))
    df = df.reindex(dates)
    df.index.rename('date', inplace=True)
    return df


def reindex_by_day(df):
    """
        Assumes data is already sorted in the right sequency.
        Reindexes the data by daily date. So, if a day was missing, now it will be added and have nan value
        :param df: a pandas dataframe with columns: datetime, site and value
        :return: a pandas dataframe with columns: datetime, site and value. However, it reindex the data by date and fills missing data (because the date didnt exist) for other columns with NaN value
    """
    global start_year, end_year
    logger.info('Reindexing by full date range')

    dates = pd.to_datetime(pd.date_range(start_year + '-01-01', end_year + '-12-31', freq='D').date) # keeps only the date part
    df = df.reindex(dates)
    df.index.rename('date', inplace=True)
    return df


def separate_site(df, parameter):
    """
        Separates each site into a new column (sites were inside one single column, now they get separated, one column per site)
        :param df: a pandas dataframe with columns: datetime, site and value indexed by [sitenum, date].
        :return: new dataframe with date column and one column for each site named: parameter_sitenumber (size varies since depends on the number of sites).
    """
    logger.info('Separating the sites into new columns')

    # 'unstacks' the site values from the site column and creates the new columns
    new_df = df.unstack(level=1)
    # need to 'droplevel' because before that, each site was a sub column of a newly created column
    new_df.columns = new_df.columns.droplevel()
    # renames each column to contain the parameter as the prefix
    new_df.columns = [str(parameter) + '_' + str(col) for col in new_df.columns]

    return new_df


def calc_missing_daily(row):
    """
        Calculates the percentage of missing values
        :param row: one row with the sum of all null values
    """
    global start_year, end_year

    num_days = ((float(end_year) - float(start_year)) + 1) * 365
    return (float(row)/num_days)*100

def calc_missing_8h(row):
    """
        Calculates the percentage of missing values
        :param row: one row with the sum of all null values
    """
    global start_year, end_year

    num_days = ((float(end_year) - float(start_year)) + 1) * 365 * 3
    return (float(row)/num_days)*100


def calc_stats_8h(df, parameter):
    """
        Calculates and prints the percentage of missing values in each column
        :param df: dataframe with columns of data
        :param parameter: number
    """
    logger.info('Calculating statistcs about the columns')
    missing = df.isnull().sum().to_frame('missing').apply(calc_missing_8h, axis=1)

    print 'Number of stations for parameter: ' + str(parameter) + ' is: ' + str(len(df.columns))
    print 'PARAM_SITE  PERCENTAGEMISSING'
    print missing

def calc_stats_daily(df, parameter):
    """
        Calculates and prints the percentage of missing values in each column
        :param df: dataframe with columns of data
        :param parameter: number
    """
    logger.info('Calculating statistcs about the columns')
    missing = df.isnull().sum().to_frame('missing').apply(calc_missing_daily, axis=1)

    print 'Number of stations for parameter: ' + str(parameter) + ' is: ' + str(len(df.columns))
    print 'PARAM_SITE  PERCENTAGEMISSING'
    print missing


def remove_other_site_cols(df, site):
    for col in df.columns:
        if col.split('_')[1] != site:
            del df[col]

# --- START clean
@utils.timeit
def clean_data_8h(p_start_year, p_end_year, county, clean_output_path, site, state='48'):
    """
    """
    logger.info('Started MAIN')
    # temp_parameter = '42401'
    # selecting folder:
    global start_year, end_year
    start_year = p_start_year
    end_year = p_end_year
    root_dir = str(state +'/' + county + '/' + start_year + '-'+ end_year + '/')
    # calc_max = ['44201']

    # iterate over files inside folder
    for filename in os.listdir(root_dir):
        complete_path = os.path.join(root_dir, filename)

        df = pd.read_csv(complete_path, skipfooter=1, engine='python')

        if df.empty:
            logger.warning('DataFrame with file: %s is empty. Continue to the next param', complete_path)
            continue

        parameter = filename.split('.')[0] # sets the parameter

        # if parameter != temp_parameter: #### TEMPORARY FOR TESTING
        # 	logger.warning('Using temp_parameter: %s . Continue to the next param', temp_parameter)
        # 	continue ### TEMPORARY FOR TESTING

        logger.info('DO:DataFrame in file: %s will be modified', complete_path)

        # --------- drop columns
        df = df[['site','value','datetime']] # only keeps the relevant columns
        # --------- substitute de values from columns datetime and site, becomes date and site
        df = clean_datetime_site_8h(df)
        # --------- calculates 8h average TODO:implement to select maximum or average
        # df = calc_8h_average(df)
        df = calc_8h_maximum(df)
        # --------- separates the site column into new features
        df = separate_site(df, parameter)
        # --------- reindex the date and fills with the full range.
        # df = reindex_by_date(df)
        df = reindex_by_8h(df)
        # --------- deletes columns form other sites TODO: this should be optional. For now every other column besides the one with 0069 is being deleted. Cannot delete before and save computational time because data is mixed in the dataframe.
        if site:
            remove_other_site_cols(df, site)
        # --------- calculate statistics about the data
        calc_stats_8h(df, parameter)
        # --------- writes file to new folder with prefix
        utils.write_new_csv(df, clean_output_path, filename, county, state, start_year, end_year)

        logger.info('DONE:DataFrame in file: %s was modified', complete_path)


    logger.info('Finished MAIN')

def clean_data_daily(p_start_year, p_end_year, county, clean_output_path, site, state='48'):
    """
        """
    logger.info('Started MAIN')
    # temp_parameter = '42401'
    # selecting folder:
    global start_year, end_year
    start_year = p_start_year
    end_year = p_end_year
    root_dir = str(
        state + '/' + county + '/' + start_year + '-' + end_year + '/')
    # calc_max = ['44201']

    # iterate over files inside folder
    for filename in os.listdir(root_dir):
        complete_path = os.path.join(root_dir, filename)

        df = pd.read_csv(complete_path, skipfooter=1, engine='python')

        if df.empty:
            logger.warning(
                'DataFrame with file: %s is empty. Continue to the next param',
                complete_path)
            continue

        parameter = filename.split('.')[0]  # sets the parameter

        # if parameter != temp_parameter: #### TEMPORARY FOR TESTING
        # 	logger.warning('Using temp_parameter: %s . Continue to the next param', temp_parameter)
        # 	continue ### TEMPORARY FOR TESTING

        logger.info('DO:DataFrame in file: %s will be modified', complete_path)

        # --------- drop columns
        df = df[
            ['site', 'value', 'datetime']]  # only keeps the relevant columns
        # --------- substitute de values from columns datetime and site, becomes date and site
        df = clean_datetime_site_daily(df)
        # --------- calculates 8h average TODO:implement to select maximum or average
        # df = calc_8h_average(df)
        df = calc_daily_max(df)
        # --------- separates the site column into new features
        df = separate_site(df, parameter)
        # --------- reindex the date and fills with the full range.
        # df = reindex_by_date(df)
        df = reindex_by_day(df)
        # --------- deletes columns form other sites TODO: this should be optional. For now every other column besides the one with 0069 is being deleted. Cannot delete before and save computational time because data is mixed in the dataframe.
        if site:
            remove_other_site_cols(df, site)
        # --------- calculate statistics about the data
        calc_stats_daily(df, parameter)
        # --------- writes file to new folder with prefix
        utils.write_new_csv(df, clean_output_path, filename, county, state,
                            start_year, end_year)

        logger.info('DONE:DataFrame in file: %s was modified', complete_path)

    logger.info('Finished MAIN')