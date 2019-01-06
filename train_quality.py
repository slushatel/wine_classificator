from sklearn.model_selection import train_test_split, StratifiedKFold
import numpy as np
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
import os
import keras.models
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score


class TrainQuality:
    def split_to_train_test_sets(self, wines):
        y = wines.quality
        X = wines.drop('quality', axis=1)
        X = StandardScaler().fit_transform(X)
        return X, y

    def get_model(self, X, Y):
        model_path = './wine_model_quality.h5'
        print(os.path.abspath(model_path))
        if os.path.isfile(model_path):
            model = keras.models.load_model(model_path);
        else:
            model = Sequential()
            model.add(Dense(64, input_dim=12, activation='relu'))
            model.add(Dense(1))
            seed = 7
            np.random.seed(seed)
            kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            for train, test in kfold.split(X, Y):
                model = Sequential()
                model.add(Dense(64, input_dim=12, activation='relu'))
                model.add(Dense(1))
                model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
                model.fit(X[train], Y[train], epochs=10, verbose=1)
            model.save(model_path)
        return model

    def train(self, X_train, y_train):
        model = self.get_model(X_train, y_train)
        return model

    def calc_additional(self, y_test, y_pred):
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print("Confusion matrix: " + str(cm))

        # Precision
        ps = precision_score(y_test, y_pred)
        print("Precision: " + str(ps))

        # Recall
        rs = recall_score(y_test, y_pred)
        print("Recall: " + str(rs))

        # F1 score
        f1s = f1_score(y_test, y_pred)
        print("F1 score: " + str(f1s))

        # Cohen's kappa
        cks = cohen_kappa_score(y_test, y_pred)
        print("Cohen's kappa: " + str(cks))
