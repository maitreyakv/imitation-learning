"""
Copyright (C) 2020 Maitreya Venkataswamy - All Rights Reserved
"""

__author__ = "Maitreya Venkataswamy"

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR, LinearSVR
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from tqdm import tqdm

def main():
    df = pd.read_csv("data.csv")

    X = df.drop("u", axis="columns")
    y = df.drop([col for col in df.columns if not "u" in col], axis="columns")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    pipe = Pipeline([('scaler', StandardScaler()),
                     ('nys',    Nystroem(random_state=0, n_components=100)),
                     ('svr',    LinearSVR(max_iter=10000, random_state=0)),
                    ])

    params = {'svr__C':np.logspace(-3, 3, 10)} #, 'epsilon':np.logspace(-3, 3, 10)}
    model = GridSearchCV(pipe, param_grid=params, verbose=99,
                        n_jobs=-1, return_train_score=True).fit(X_train, y_train.to_numpy().ravel())
    plt.figure()
    plt.semilogx(model.cv_results_["param_svr__C"], model.cv_results_["mean_test_score"])
    plt.semilogx(model.cv_results_["param_svr__C"], model.cv_results_["mean_train_score"])

    print(model.score(X_test, y_test.to_numpy().ravel()))
    plt.figure()
    plt.scatter(y_test, model.predict(X_test), s=10, alpha=1., marker=".")
    plt.plot([np.amin(y_test), np.amax(y_test)], [np.amin(y_test), np.amax(y_test)], "--r")

    plt.show()


if __name__ == "__main__":
    main()
