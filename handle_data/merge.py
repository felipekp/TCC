"""
    This file contains a method to merge all files from a folder into a single file maintaining only one index.
"""

# --- general imports
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import utils.utils as utils

pd.set_option('display.max_rows', 200000)

# --- global variables
global start_year
global end_year

# --- Start CODE

@utils.timeit
def merge_data(p_start_year, p_end_year, county, merge_input_path, merge_output_path, state='48', site='0069'):
    # to select folder:
    global start_year, end_year
    start_year = p_start_year
    end_year = p_end_year
    root_dir = str(state + '/'+ county + '/' + merge_input_path + start_year + '-' + end_year + '/')

    # concatenates each file and resets the index to: 0, 1, 2, 3 ...
    df = pd.concat((pd.read_csv(os.path.join(root_dir, f)) for f in os.listdir(
        root_dir)), axis=1, join='outer').set_index('date').reset_index(drop=True)

    # print df
    df.to_csv(merge_output_path + start_year + '-' + end_year + '.csv')


# if __name__ == "__main__":
#     main()
