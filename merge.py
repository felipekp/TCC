# --- general imports
import pandas as pd
import os

pd.set_option('display.max_rows', 200000) 

# --- Start CODE
def reindex_by_date(df, start_year, end_year):
	'''
		Reindexes the data by daily data! So, if a day was missing now it will be added and have nan value
	'''
	dates = pd.to_datetime(pd.date_range(start_year + '-01-01', end_year + '-12-31', freq='D').date) # keeps only the date part
	
	df['date'] = dates
	df = df.set_index('date')
	
	return df

def main():
	# to select folder:
	start_year = '2000'
	end_year = '2016'
	root_dir = str('48/113/refine-' + start_year + '-'+ end_year + '/')

	# concatenates each file and resets the index to: 0, 1, 2, 3 ...
	df = pd.concat((pd.read_csv(os.path.join(root_dir, f)) for f in os.listdir(root_dir)), axis=1, join='outer').set_index('date').reset_index(drop=True)#

	# indexed as 1, 2, 3... but will be reindexed as: 2013-01-01, 2013-01-02, 2013-01-03 again...
	# df = reindex_by_date(df, start_year, end_year)

	# print df
	df.to_csv('merged_' + start_year + '-' + end_year + '.csv')


if __name__ == "__main__":
    main()

