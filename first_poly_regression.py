import numpy as np
import datetime as DT
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from pandas import DataFrame
from scipy import stats
from sklearn.metrics import r2_score
import time

my_file = open('48/029/2013/44201.csv')

# verification of the file

df = pd.read_csv(my_file)

# unique for a site for each day... we start separating the data here base on this variable
county_df = df.groupby(['site', 'lat', 'lon', 'datetime'])

sitenum = list()
datetime = list()
mean_meas = list()

for uniqgrp_name,grp_data in county_df:
	sitenum.append(uniqgrp_name[0]) # unique site
	datetime.append(uniqgrp_name[3][0:8]) # ignores the timezone part and appends the datetime (organizing data to be ploted in a timeline)
	mean_meas.append(grp_data['value'].mean()) # calculates the mean value and appends it

sites_dict = {'SiteNum': sitenum, 'Datetime': datetime,'MeanOzone': mean_meas}
sites_df = DataFrame(sites_dict)

new_group = sites_df.groupby(['SiteNum', 'Datetime'])

new_datetime = list()
new_mean1 = list()

for x, y in new_group:
	## They all need separated new_datetime because some do not have data for some days.....
	# 840480290059 840480290032 840480290052
	if x[0] == '840480290032':
		# separates the datetime field to be plotted
		temp_new_date = DT.datetime(int(x[1][0:4]), int(x[1][4:6]), int(x[1][6:8]))
		# appends the new datetime to be used on the X axis
		new_datetime.append(temp_new_date)
		# the mean value calculated for that day is appended to be used in the Y axis
		new_mean1.append(y['MeanOzone'].mean())

# new stuff for fit
new_datetime = np.array(new_datetime)
new_mean1 = np.array(new_mean1)

degress = [1,2,4,8,20]

for fitting_degree in degress:
	y = new_mean1
	x = mdates.date2num(new_datetime)

	p4 = np.poly1d(np.polyfit(x, y, fitting_degree))

	xp = np.linspace(x.min(), x.max(), 100)

	plt.plot(x, y, '.', label='blub') # prints the data
	plt.plot(mdates.num2date(xp), p4(xp), c='r') # prints the fit as a red line

	r2 = r2_score(y, p4(x))
	print fitting_degree, r2

	plt.show()

	plt.savefig('fit_' + str(fitting_degree) + '.png')
