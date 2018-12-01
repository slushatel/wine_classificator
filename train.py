from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler


class Trainer:
    def split_to_train_test_sets(self, wines):
        # Specify the data
        X = wines.ix[:, 0:11]

        # Specify the target labels and flatten the array
        y = np.ravel(wines.type)

        # Split the data up in train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        return X_train, X_test, y_train, y_test

    def standartize_data(self, X_train, X_test):
        # Define the scaler
        scaler = StandardScaler().fit(X_train)

        # Scale the train set
        X_train = scaler.transform(X_train)

        # Scale the test set
        X_test = scaler.transform(X_test)

        return X_train, X_test

