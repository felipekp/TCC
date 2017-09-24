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
              (method.__name__, te-ts)
        return result

    return timed

# --- Start CODE
def calc_stats(df):
	'''
		Need these stats to know which method to apply
		large or small spaced.
	'''
	logging.info('Calculating statistcs about the columns')
	# stats = df.isnull().sum().to_frame('missing')

	# return stats


def write_new_csv(df, filename, county, start_year, end_year):
	logging.info('Saving file into new folder')
	newpath = '48/' + county + '/refine-'+ start_year + '-'+ end_year
	if not os.path.exists(newpath):
		os.makedirs(newpath)

	df.to_csv(os.path.join(newpath, filename))

def handle_outliers(df):
	'''
		There are no silver bullets for this issue...
		Removes negative values from readings that should not have negative values
	'''
	logging.warning('Removing outliers (negative values and putting 0)')

	df = df.clip_lower(0)

	return df

def normalize_scaller(df):
	pass

# def verify_lowerzero(df, parameter):
# 	for x in (df < 0).any(): 
# 		if x == True:
# 			print parameter

def calc_dispersion_missing(df):
	'''
		Calculates the dispersion of missing data
	'''
	new_df = df.isnull().astype(int).groupby(df.notnull().astype(int).cumsum()).sum()

	# for item in new_df:
	# 	if item > 2000:
	# 		print item

	# maximum consecutive days without readings
	if new_df.max().astype(int) > 365:
		# print new_df.max()
		logger.critical('Cannot interpolate using linear method because max size is over 365')
		return df, True
	logger.warning('SMALL. Maximum size is under one year then we can interpolate some parts using a 30 days limit for linear')
	df = df.interpolate(limit_direction='both', limit=30)
	
	# df = df.fillna()
	logger.warning('MEDIUM. Using mean from previous and next year to fill this gap')
	window = 2 # size of window to look for value
	gap = 365 # for year gap (takes values from next and previous year) (if it was 30, would use values from previous and next months)
	df[df.isnull()] = np.nanmean([df.shift(x).values for x in range(-gap*window,gap*(window+1),gap)], axis=0 )

	new_df = df.isnull().astype(int).groupby(df.notnull().astype(int).cumsum()).sum()

	# print new_df.max()
	# print df


	return df, False

def workaround_interpolate(df, parameter):
	'''
		One idea: based on this: https://stackoverflow.com/questions/32850185/change-value-if-consecutive-number-of-certain-condition-is-achieved-in-pandas
	'''
	for col in df.columns:
		df[col], delete_col = calc_dispersion_missing(df[col])
		if delete_col:
			del df[col]
	return df

def handle_interpolate(df, parameter):
	'''
		
	'''
	#classify every collumn into: large or small gap (small will be handled with interpolate and large not)
	logger.info('Handling interpolate')

	return workaround_interpolate(df, parameter)

	# for col in df.columns:
	# 	print col
	# 	if calc_dispersion_missing(df[col]) < 0.3:
	# 		print 'small'
	# 	else:
	# 		print 'large'

	# 	print '-----'

	# return df


@timeit
def main():
	logging.info('Started MAIN')
	# temp_parameter = '43205'
	# to select folder:
	start_year = '2000'
	end_year = '2016'
	county = '113'
	root_dir = str('48/' + county + '/clean-' + start_year + '-'+ end_year + '/')
	not_remove_param_outlier = ['68105'] # parameters that will not go thought outlier removal

	# iterate over files inside a folder
	for filename in os.listdir(root_dir):
		complete_path = os.path.join(root_dir, filename)

		df = pd.read_csv(complete_path, skipfooter=1, engine='python')
		df = df.set_index(['date'])

		if df.empty:
			logger.warning('DataFrame with file: %s is empty. Passing', complete_path)
			continue

		parameter = filename.split('.')[0] # sets the parameter

		# if parameter != temp_parameter: #### TEMPORARY FOR TESTING
		# 	continue ### TEMPORARY FOR TESTING

		logger.info('DO:DataFrame in file: %s will be modified', complete_path)

		# --------- handling outliers:
		if parameter not in not_remove_param_outlier:
			df = handle_outliers(df)
		# --------- handling interpolate:
		df = handle_interpolate(df, parameter)
		# --------- writes file to new folder with prefix: 'clean-'
		write_new_csv(df, filename, county, start_year, end_year)

		logger.info('DONE:DataFrame in file: %s was modified', complete_path)


	logging.info('Finished MAIN')


if __name__ == "__main__":
    main()

