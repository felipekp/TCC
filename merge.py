# --- general imports
import numpy as np
import pandas as pd
import matplotlib as plt
import datetime as DT
import os

pd.set_option('display.max_rows', 200000) 

# --- Start CODE
def reindex_by_date(df, start_year, end_year):
	'''
		Reindexes the data by daily data! So, if a day was missing now it will be added and have nan value
	'''
	# TODO: check if I have to use df.sort_values(['datetime']) before reindexing...
	dates = pd.to_datetime(pd.date_range(start_year + '-01-01', end_year + '-12-31', freq='D').date) # keeps only the date part
	
	df['date'] = dates
	df = df.set_index('date')
	
	return df

def main():
	# temp_parameter = '43860'
	# to select folder:
	start_year = '2013'
	end_year = '2013'
	root_dir = str('48/029/clean-' + start_year + '-'+ end_year + '/')

	# iterate over files inside a folder
	df = pd.concat((pd.read_csv(os.path.join(root_dir, f)) for f in os.listdir(root_dir)), axis=1, join='outer').set_index('date').reset_index(drop=True)#

	# not really necessary, but good for visualization
	df = reindex_by_date(df, start_year, end_year)

	# print df
	df.to_csv('out.csv')


if __name__ == "__main__":
    main()

