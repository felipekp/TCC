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


def handle_outliers(df):
	'''
		MUST REMOVE NEGATIVE VALUES??
		ALSO! CANT REMOVE THE 95 PERCENTILE.. IT IS RELEVANT FOR THE DATA..
		BUT ITS KIND OF HANDLED WITH DAILY MEAN WHEN THERE ARE SEVERAL READING FOR ONE DAY.

		Data must be on a scale of 0 to 1 already
	'''
	logging.critical('NOT IMPLEMENTED!!! EXIT')
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

@timeit
def main():
	logging.info('Started MAIN')
	temp_parameter = '43860'
	# to select folder:
	start_year = '2013'
	end_year = '2013'
	root_dir = str('48/029/clean-' + start_year + '-'+ end_year + '/')

	# iterate over files inside a folder
	for filename in os.listdir(root_dir):
		complete_path = os.path.join(root_dir, filename)

		df = pd.read_csv(complete_path, skipfooter=1, engine='python')

		if df.empty:
			logger.warning('DataFrame with file: %s is empty. Passing', complete_path)
			continue

		parameter = filename.split('.')[0] # sets the parameter
		if parameter != temp_parameter: #### TEMPORARY FOR TESTING
			continue ### TEMPORARY FOR TESTING

		logger.info('DO:DataFrame in file: %s will be modified', complete_path)

		# # --------- handling outliers:
		df = handle_outliers(df)
		# --------- calculate statistics about the data
		# calc_stats(df)
		# --------- writes file to new folder with prefix: 'clean-'
		# write_new_csv(df, filename, start_year, end_year)

		logger.info('DONE:DataFrame in file: %s was modified', complete_path)


	logging.info('Finished MAIN')


if __name__ == "__main__":
    main()

