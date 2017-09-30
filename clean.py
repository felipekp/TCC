# --- general imports
import datetime as DT
import matplotlib as plt
import numpy as np
import os
import pandas as pd

pd.set_option('display.max_rows', 200000) # so pandas prints more rows

# --- logging
import logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(funcName)20s() %(levelname)-8s %(message)s',
                    datefmt='%d-%m %H:%M:%S',
                    filename='clean.log',
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
              (method.__name__, te-ts)
        return result

    return timed

# --- global variables
# global variables were used because of .apply() function used inside calc_stats, otherwise I wouldve used these two as normal parameters. They receive value inside 'main()'
global start_year
global end_year

# --- START Functions
def clean_datetime_site(df):
	"""
		Clears data in datetime and site field of a given pandas dataframe.
		site becomes: 'site' containing only the site code
		datetime becomes: 'date' containing only year-month-day
		value is repeated.
		:param df: a pandas dataframe with columns: datetime, site and value
		:return: a modified dataframe with columns: date, site and value
	"""
	logging.info('Creating new DataFrame with site, date and repeating the value')

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


def calc_daily_mean(df):
	"""
		Calculates daily mean and reindexes the data by (multiple indexes) date and sitenum (site was renamed to sitenum here)
		:param df: a pandas dataframe with columns: datetime, site and value
		:return: pandas dataframe reindexed with date and sitenum and daily mean in the value column
	"""
	logging.info('Calculating daily MEAN and index by date and site')

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
	logging.info('Calculating daily MAX and index by date and sitenum')

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


def reindex_by_date(df):
	"""
		Assumes data is already sorted in the right sequency.
		Reindexes the data by daily date. So, if a day was missing, now it will be added and have nan value
		:param df: a pandas dataframe with columns: datetime, site and value
		:return: a pandas dataframe with columns: datetime, site and value. However, it reindex the data by date and fills missing data (because the date didnt exist) for other columns with NaN value
	"""
	global start_year, end_year
	logging.info('Reindexing by full date range')

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
	logging.info('Separating the sites into new columns')
	
	# 'unstacks' the site values from the site column and creates the new columns
	new_df = df.unstack(level=1)
	# need to 'droplevel' because before that, each site was a sub column of a newly created column
	new_df.columns = new_df.columns.droplevel()
	# renames each column to contain the parameter as the prefix
	new_df.columns = [str(parameter) + '_' + str(col) for col in new_df.columns]
	
	return new_df


def calc_missing(row):
	"""
		Calculates the percentage of missing values
		:param row: one row with the sum of all null values
	"""
	global start_year, end_year

	num_days = ((float(end_year) - float(start_year)) + 1) * 365
	return (float(row)/num_days)*100


def calc_stats(df, parameter):
	"""
		Calculates and prints the percentage of missing values in each column
		:param df: dataframe with columns of data
		:param parameter: number
	"""
	logging.info('Calculating statistcs about the columns')
	missing = df.isnull().sum().to_frame('missing').apply(calc_missing, axis=1)

	print 'Number of stations for parameter: ' + str(parameter) + ' is: ' + str(len(df.columns))
	print 'PARAM_SITE  PERCENTAGEMISSING'
	print missing


def write_new_csv(df, filename, county):
	"""
		Saves the dataframe inside a new file in a new path (a folder with 'clean-' as prefix)
		:param df: dataframe with the modified data
		:param filename: filename from file being read (file name will stay the same)
		:param county: county number
	"""
	global start_year, end_year
	logging.info('Saving file into new folder')
	newpath = '48/' + county + '/max-clean-'+ start_year + '-'+ end_year
	if not os.path.exists(newpath):
		os.makedirs(newpath)

	df.to_csv(os.path.join(newpath, filename))

# --- END FUNCTIONS
# --- START Main
@timeit
def main():
	logging.info('Started MAIN')
	# temp_parameter = '42401'
	# selecting folder:
	global start_year, end_year
	start_year = '2000'
	end_year = '2016'
	county = '113'
	root_dir = str('48/' + county + '/' + start_year + '-'+ end_year + '/')
	calc_max = ['44201']

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
		df = clean_datetime_site(df)
		# --------- calculates the mean for each day
		if parameter in calc_max:
			df = calc_daily_max(df)
		else:
			df = calc_daily_mean(df)
		# --------- separates the site column into new features
		df = separate_site(df, parameter)
		# --------- reindex the date and fills with the full range. 
		df = reindex_by_date(df)
		# --------- calculate statistics about the data
		calc_stats(df, parameter)
		# --------- writes file to new folder with prefix: 'clean-'
		write_new_csv(df, filename, county)
		
		logger.info('DONE:DataFrame in file: %s was modified', complete_path)


	logging.info('Finished MAIN')


if __name__ == "__main__":
    main()

