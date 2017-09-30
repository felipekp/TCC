"""
source code: https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
"""
from pandas import read_csv
from matplotlib import pyplot
# load dataset
start_year = '2000'
end_year = '2016'
site = '0069'

data_file_name = 'merged/max-merged_' + start_year + '-' + end_year + '.csv'

dataset = read_csv(data_file_name, header=0, index_col=0)
values = dataset.values
# specify columns to plot
groups = [num for num in range(len(dataset.columns))]
i = 1
# plot each column
pyplot.figure()
for group in groups:
    pyplot.subplot(len(groups), 1, i)
    pyplot.plot(values[:, group])
    pyplot.title(dataset.columns[group], y=0.5, loc='right')
    i += 1
pyplot.show()