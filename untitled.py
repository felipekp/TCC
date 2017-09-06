@timeit
def verify_uniqueness(df, col_name):
	### can make this more generic??
	global uniq_col

	temp_unit = df[col_name].unique() # this might be too time consuming because it should stop if found one different value. However, it returns the entire list for all unique vals (can try to avoid this by changing the function to: 'duplicated()', then retrieving one value of the column from any row).

	if len(temp_unit) == 1:
		uniq_col[col_name] = temp_unit[0]
		logging.info('Only one value in the column name: ' + str(col_name))
		del df[col_name]
	else:
		logging.info('Different than one on the col: ' + str(col_name))


def clean_datetime(df):
	'''
		Replaces the datetime column with only year-month-day in pandas timestamp format.
	'''
	new_df = df.groupby('datetime')

	for leftf, rightf in new_df:
		# uses pandas timestamp to hold the data now.
		temp_new_date = pd.Timestamp(DT.datetime(int(str(leftf)[0:4]), int(str(leftf)[4:6]), int(str(leftf)[6:8]))) # fields are: year, month, day, hour

		df['datetime'] = df['datetime'].replace(leftf, temp_new_date) # REPLACES the current datetime field (kind of not as efficient as I wished)

		# possible solution that might help: calculate the mean ozone for each day beforehand (maybe even for each site too.. also, will have to change the groupby to use site too, like: new_df = df.groupby(['site', 'datetime'])), 