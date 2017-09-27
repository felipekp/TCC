# --- general imports
import pandas as pd
import os

pd.set_option('display.max_rows', 200000) 

# --- global variables
global start_year
global end_year

# --- Start CODE
def main():
	# to select folder:
	global start_year, end_year
	start_year = '2000'
	end_year = '2016'
	root_dir = str('48/113/refine-' + start_year + '-'+ end_year + '/')

	# concatenates each file and resets the index to: 0, 1, 2, 3 ...
	df = pd.concat((pd.read_csv(os.path.join(root_dir, f)) for f in os.listdir(root_dir)), axis=1, join='outer').set_index('date').reset_index(drop=True)

	# print df
	df.to_csv('merged/merged_' + start_year + '-' + end_year + '.csv')


if __name__ == "__main__":
    main()

