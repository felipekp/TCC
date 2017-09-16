# --- general imports
import numpy as np
import pandas as pd
import matplotlib as plt
import datetime as DT

pd.set_option('display.max_rows', 200000) 

# --- logging
import logging
logging.basicConfig(filename='data_clean.log', filemode='w', level=logging.INFO)

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
global state
global county

global uniq_cols
global safe_cols
safe_cols = ['site', 'value', 'datetime']
uniq_cols = {}

# @timeit
def verify_uniqueness(df, col_name):
	### can make this more generic??
	global uniq_cols, safe_cols

	if col_name in safe_cols: 
		logging.info('Safe name. Not going to check for uniqueness: ' + str(col_name))
		return

	temp_unit = df[col_name].unique() # this might be too time consuming because it should stop if found one different value. However, it returns the entire list for all unique vals (can try to avoid this by changing the function to: 'duplicated()', then retrieving one value of the column from any row).

	if len(temp_unit) == 1:
		uniq_cols[col_name] = temp_unit[0]
		logging.info('Only one value in the column name: ' + str(col_name))
		del df[col_name]
	else:
		logging.info('Different than one in the col: ' + str(col_name))

def drop_sameval_columns(df):
	'''
		For every column, verifies if all values in it are the same. If all the same, save in a global var the value and drop the column, else continue to the next column.
			Maybe add: Drops rows that are known to not have relevant info (like latitute that might not get dropped if there are more than 1 site)
	'''
	for col_name in df.columns.values:
		verify_uniqueness(df, col_name)

def clean_site(df):
	'''
		Removes country, state and county from site column. Only leaves the site number on the column and stores state and county in global variables.
	'''
	global state, county

	new_df = df.groupby('site')
	
	all_counties = set()
	all_states = set()
	
	for leftf, rightf in new_df:
		df['site'] = df['site'].replace(leftf, str(leftf)[8:]) # REPLACES the whole column values with only the site number. TODO: measure the time this consumes. This might not be as efficient as other approaches.
		all_states.add(str(leftf)[3:5])
		all_counties.add(str(leftf)[5:8])

	if len(all_states) != 1 or len(all_counties) != 1: # verifies if there is indeed only one state and one county in the entire file
		print 'Error.. more than one county or state here!!'
	else:
		# saves state and county in global variables.
		state = all_states.pop()
		county = all_counties.pop()
		
	# del df['site']

def temp_drop_columns(df):
	logging.info('Dropping columns for the main file')
	del df['parameter']
	del df['unit']
	del df['method_code']
	del df['qc']
	del df['poc']
	del df['lat']
	del df['lon']
	del df['GISDatum']
	del df['elev']
	del df['mpc']
	del df['mpc_value']
	del df['data_status']
	del df['action_code']
	del df['duration']
	del df['uncertainty']
	del df['qualifiers']
	del df['frequency']

def clean_datetime_site(df):
	'''
		Clears data in the datetime and site field.
		site becomes: 'site' containing only the site code
		datetime becomes: 'date' containing only year-month-day
	'''
	logging.info('Creating new DataFrame with only site (filters), date (filters) and repeating the value')
	new_df = df.groupby(['site', 'datetime'])
	date = list()
	sitenum = list()

	for leftf, rightf in new_df:
		sitenum.append(str(leftf[0])[8:])
		date.append(pd.Timestamp(DT.datetime(int(str(leftf[1])[0:4]), int(str(leftf[1])[4:6]), int(str(leftf[1])[6:8]))))


	return pd.DataFrame({'site': sitenum, 'date': date, 'value': df.value})

def calc_daily_mean(df):
	new_df = df.groupby(['site', 'date'])
	calc_mean = list()
	date = list()
	sitenum = list()

	logging.info('Calculating daily mean and index by date')

	for leftf, rightf in new_df:
		calc_mean.append(rightf['value'].mean())
		sitenum.append(leftf[0])
		date.append(leftf[1])

	return pd.DataFrame({'site': sitenum, 'dailyMean': calc_mean}, index=date)

def reindex_by_date(df, start_year, end_year):
	logging.info('Reindexing by full date range')
	print start_year, end_year
	# TODO: check if I have to sue df.sort_values(['datetime']) before reindexing...
	dates = pd.to_datetime(pd.date_range(start_year +'-01', end_year + '-12-31', freq='D').date) # keeps only the date part
	return df.reindex(dates)

def fill_site_column(df):
	df['site'] = df['site'].ffill() # forward fill
	df['site'] = df['site'].bfill() # backward fill
	return df

def handle_outliers(df):
	'''
		MUST REMOVE NEGATIVE VALUES??

		Data must be on a scale of 0 to 1 already
	'''
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
	parameter = '44201'
	start_year = '2013'
	end_year = '2013'

	my_file = open('48/029/' + start_year + '-'+ end_year +'/' + parameter + '.csv')

	# put the data inside a pandas dataframe
	df = pd.read_csv(my_file, skipfooter=1, engine='python')
	# --------- drop columns
	df = df[['site','value','datetime']]# temp_drop_columns(df)
	# --------- substitute de values from columns datetime and site, becomes date and site
	df = clean_datetime_site(df)
	# --------- handling outliers:
	# df = handle_outliers(df)
	print df['value'].value_counts()
	exit()
	# --------- calculates the mean for each day
	df = calc_daily_mean(df)
	# --------- reindex the date and fills with the full range
	print 'number of rows BEFORE reindex: ', len(df.index)
	new_df = df.groupby('site').apply(reindex_by_date, start_year, end_year).reset_index(0, drop=True)
	print 'number of rows AFTER reindex: ', len(new_df.index)
	# --------- changes the order the columns are displayed
	new_df = new_df[['site', 'dailyMean']]
	# --------- fills the site column with the right site number
	fill_site_column(new_df)
	
	# --------- interpolate method
	print 'missing data BEFORE interpolate: ', len(new_df[new_df.isnull().any(axis=1)])
	new_df['dailyMean'] = new_df['dailyMean'].interpolate(method='time') # interpolates the data
	print 'missing data AFTER interpolate: ', len(new_df[new_df.isnull().any(axis=1)])

	# --------- renames the index column to: 'date' (no name before)
	new_df.index.rename('date', inplace=True)
	# --------- writes to csv
	new_df.to_csv(parameter + '.out')

	logging.info('Finished MAIN')


if __name__ == "__main__":
    main()

