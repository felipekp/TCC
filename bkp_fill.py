# --- general imports
import numpy as np
import pandas as pd
import matplotlib as plt
import datetime as DT

# --- logging
import logging
logging.basicConfig(filename='date_clean.log', filemode='w', level=logging.INFO)

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
		logging.info('Different than one on the col: ' + str(col_name))

	

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

def handle_missing_days(df):
	'''

	'''
	new_df = df.groupby('site')

	for leftf, rightf in new_df:
		print leftf

def temp_drop_columns(df):
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

	for leftf, rightf in new_df:
		calc_mean.append(rightf['value'].mean())
		sitenum.append(leftf[0])
		date.append(leftf[1])

	return pd.DataFrame({'site': sitenum, 'date': date, 'dailyMean': calc_mean}, index=date)

def reindex_by_date(df):
    dates = pd.date_range('2013-01', '2013-12-31', freq='D').date # keeps only the date part
    return df.reindex(dates).ffill()

def main():
	logging.info('Started MAIN')
	my_file = open('48/029/2013/44201.csv')

	# put the data inside a pandas dataframe
	df = pd.read_csv(my_file, skipfooter=1, engine='python')
	# --------- drop columns
	temp_drop_columns(df)
	# --------- substitute de values from columns datetime and site, becomes date and site
	df = clean_datetime_site(df)
	# --------- calculates the mean for each day
	df = calc_daily_mean(df)

	# print df.head()

	new_df = df.groupby('site').apply(reindex_by_date).reset_index(0, drop=True)
	# dates = pd.date_range('2013-01', '2013-12-31', freq='D').date # keeps only the date part
	# dates_df = pd.DataFrame({'site': np.NaN, 'date': dates, 'meanValue': np.NaN})

	# print dates_df.head()

	print new_df

	# for leftf, rightf in site_df:
	# 	# aqui criar novos negocios e copiar os antigos em novas listas
	# 	print rightf

	# print new_df.head()

	# print df.head()

	exit()




	# drop_sameval_columns(df)
	# print df.columns to list all the columns on the dataframe
	# gives a pretty good idea about the data. Including 25, 50 and 75 percentile.
	# print df.describe()

	# # --------- cleaning datetime
	# clean_datetime(df)

	# # --------- cleaning site
	# clean_site(df)

	# # --------- cleaning datetime
	# clean_datetime(df)


	# # --------- finding missing days and fixing it
	# handle_missing_days(df)


	print uniq_cols
	# print df.head()

	logging.info('Finished MAIN')


if __name__ == "__main__":
    main()

