# Import pandas
import pandas as pd
import visualizer as vis
import numpy as np

# Read in white wine data
white = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv", sep=';')

# Read in red wine data
red = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep=';')

# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)

# Print info on white wine
print(white.info())

# Print info on red wine
print(red.info())

# First rows of `red`
print(red.head())

# Last rows of `white`
print(white.tail())

# Take a sample of 5 rows of `red`
print(red.sample(5))

# Describe `white`
print(white.describe())

# Double check for null values in `red`
print(pd.isnull(red))

vis.Vizualizer().plot_wine_data(red, white)

print(np.histogram(red.alcohol, bins=[7,8,9,10,11,12,13,14,15]))
print(np.histogram(white.alcohol, bins=[7,8,9,10,11,12,13,14,15]))

vis.Vizualizer().plot_quality_from_sulphates(red, white)

vis.Vizualizer().plot_quality_from_volatile_acidity(red, white)

