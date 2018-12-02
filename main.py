# Import pandas
import pandas as pd
import visualizer as vis
import numpy as np
import train as train

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

# preprocess data
# Add `type` column to `red` with value 1
red['type'] = 1
# Add `type` column to `white` with value 0
white['type'] = 0
# Append `white` to `red`
wines = red.append(white, ignore_index=True)

vis.Vizualizer().plot_corelation_matrix(wines)

X_train, X_test, y_train, y_test = train.Trainer().split_to_train_test_sets(wines)

X_train, X_test = train.Trainer().standartize_data(X_train, X_test)

# Import `Sequential` from `keras.models`
from keras.models import Sequential

# Import `Dense` from `keras.layers`
from keras.layers import Dense

# Initialize the constructor
model = Sequential()

# Add an input layer
model.add(Dense(12, activation='relu', input_shape=(11,)))

# Add one hidden layer
model.add(Dense(8, activation='relu'))

# Add an output layer
model.add(Dense(1, activation='sigmoid'))

