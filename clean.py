# --- general imports
import numpy as np
import pandas as pd
import matplotlib as plt
import datetime as DT
import os

pd.set_option('display.max_rows', 200000) 

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

# --- Start CODE
def clean_datetime_site(df):
	'''
		Clears data in the datetime and site field.
		site becomes: 'site' containing only the site code
		datetime becomes: 'date' containing only year-month-day
	'''
	logging.info('Creating new DataFrame with only site, date filters and repeating the value')

	new_df = df.groupby(['site', 'datetime'])
	date = list()
	sitenum = list()

	for leftf, rightf in new_df:
		sitenum.append(str(leftf[0])[8:])
		date.append(pd.Timestamp(DT.datetime(int(str(leftf[1])[0:4]), int(str(leftf[1])[4:6]), int(str(leftf[1])[6:8]))))

	return pd.DataFrame({'site': sitenum, 'date': date, 'value': df.value})


def calc_daily_mean(df):
	'''
		Calculates the daily mean and reindex the data by (multiple indexes) date and sitenum
	'''
	logging.info('Calculating daily mean and index by date')

	new_df = df.groupby(['site', 'date'])
	calc_mean = list()
	date = list()
	sitenum = list()

	for leftf, rightf in new_df:
		calc_mean.append(rightf['value'].mean())
		sitenum.append(leftf[0])
		date.append(leftf[1])

	# multi-indexing necessary for unstacking later
	return pd.DataFrame({'dailyMean': calc_mean}, index=[date, sitenum])


def reindex_by_date(df, start_year, end_year):
	'''
	'''
	logging.info('Reindexing by full date range')
	# TODO: check if I have to use df.sort_values(['datetime']) before reindexing...
	dates = pd.to_datetime(pd.date_range(start_year + '-01-01', end_year + '-12-31', freq='D').date) # keeps only the date part
	df = df.reindex(dates)
	df.index.rename('date', inplace=True)
	return df


def handle_outliers(df):
	'''
		MUST REMOVE NEGATIVE VALUES??
		ALSO! CANT REMOVE THE 95 PERCENTILE.. IT IS RELEVANT FOR THE DATA..
		BUT ITS KIND OF HANDLED WITH DAILY MEAN WHEN THERE ARE SEVERAL READING FOR ONE DAY.

		Data must be on a scale of 0 to 1 already
	'''
	logging.info('Handling outliers')
	new_df = df.groupby(['site'])
	# TODO remove negative readings

	# identifies the 95th percentile and median for a site
	statBefore = pd.DataFrame({'median': new_df['value'].median(), 'p95': new_df['value'].quantile(.95)})
	
	print statBefore

	exit()

	for leftf, rightf in new_df:
		calc_mean.append(rightf['value'].mean())
		sitenum.append(leftf[0])
		date.append(leftf[1])
	# if row.Value > (median + (1.5* iq_range)) or row.Value < (median - (1.5* iq_range)):
	# df['outlier'] = 
	return df


def separate_site(df):
	logging.info('Separating the sites into new columns')
	
	new_df = df.unstack(level=1)
	new_df.columns = new_df.columns.droplevel()
	
	return new_df


def calc_stats(df):
	logging.info('Calculating statistcs about the columns')
	stats = df.isnull().sum().to_frame('missing')

	return stats


def write_new_csv(df, filename, start_year, end_year):
	logging.info('Saving file into new folder')
	newpath = '48/029/clean-'+ start_year + '-'+ end_year
	if not os.path.exists(newpath):
		os.makedirs(newpath)

	df.to_csv(os.path.join(newpath, filename))


@timeit
def main():
	logging.info('Started MAIN')
	# temp_parameter = '43860'
	# to select folder:
	start_year = '2013'
	end_year = '2013'
	root_dir = str('48/029/' + start_year + '-'+ end_year + '/')

	# iterate over files inside a folder
	for filename in os.listdir(root_dir):
		complete_path = os.path.join(root_dir, filename)

		df = pd.read_csv(complete_path, skipfooter=1, engine='python')

		if df.empty:
			logger.warning('DataFrame with file: %s is empty. Passing', complete_path)
			continue

		parameter = filename.split('.')[0] # sets the parameter
		# if parameter != temp_parameter: #### TEMPORARY FOR TESTING
		# 	continue ### TEMPORARY FOR TESTING

		logger.info('DO:DataFrame in file: %s will be modified', complete_path)
		# --------- drop columns
		df = df[['site','value','datetime']] # only keeps the relevant columns
		# --------- substitute de values from columns datetime and site, becomes date and site
		df = clean_datetime_site(df)
		# # --------- handling outliers:
		# df = handle_outliers(df) MAYBE HANDLE LATER
		# --------- calculates the mean for each day
		df = calc_daily_mean(df)
		# --------- separates the site column into new features
		df = separate_site(df)
		# --------- reindex the date and fills with the full range. 
		df = reindex_by_date(df, start_year, end_year)
		# --------- calculate statistics about the data
		calc_stats(df)
		# --------- writes file to new folder with prefix: 'clean-'
		write_new_csv(df, filename, start_year, end_year)
		

		logger.info('DONE:DataFrame in file: %s was modified', complete_path)


	logging.info('Finished MAIN')


if __name__ == "__main__":
    main()

