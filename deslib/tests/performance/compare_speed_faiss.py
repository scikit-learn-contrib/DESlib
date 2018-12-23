import gzip
import os
import shutil
import time
import urllib.request

import numpy as np
import pandas as pd
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split

from deslib.des.knora_e import KNORAE


def run_knorae(pool_classifiers, X_DSEL, y_DSEL, X_test, y_test, knn_type):
    knorae = KNORAE(pool_classifiers=pool_classifiers,
                    knn_classifier=knn_type)

    knorae.fit(X_DSEL, y_DSEL)

    start = time.clock()
    score = knorae.score(X_test, y_test)
    end = time.clock() - start

    return score, end


def fetch_HIGGS():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/" \
          "00280/HIGGS.csv.gz"
    if not os.path.exists("../../HIGGS.csv"):

        print("Downloading HIGGS dataset from {}".format(url))

        if not os.path.exists("../../HIGGS.gz"):
            filedata = urllib.request.urlopen(url)
            data2write = filedata.read()

            with open('../../HIGGS.gz', 'wb') as f:
                f.write(data2write)

        print("Finished downloading")
        print("Extracting HIGGS.gz")

        if not os.path.exists("../../HIGGS.csv"):
            with gzip.open('../../HIGGS.gz', 'rb') as f:
                with open('../../HIGGS.csv', 'wb') as csv_out:
                    shutil.copyfileobj(f, csv_out)

        print("Extracted csv")
    print('Reading CSV file')
    df = pd.read_csv('../../HIGGS.csv', header=None)
    data = df.values
    X = data[:, 1:]
    y = data[:, 0]

    return X, y


if __name__ == "__main__":
    rng = np.random.RandomState(123456)

    print('Preparing dataset')
    X, y = fetch_HIGGS()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                        random_state=rng)

    X_DSEL, X_train, y_DSEL, y_train = train_test_split(X_train, y_train,
                                                        test_size=0.50,
                                                        random_state=rng)
    pool_classifiers = BaggingClassifier(n_estimators=100,
                                         random_state=rng,
                                         n_jobs=-1)

    print('Fitting base classifiers...')
    pool_classifiers.fit(X_train, y_train)

    n_samples = 1000000
    num_of_test_inputs = [100, 1000, 10000]

    for n_t in num_of_test_inputs:
        print("running experiment: num_of_DSEL_samples: {}, "
              "num_of_tests: {}".format(y_DSEL.size, n_t))

        score_sklearn, time_sklearn = run_knorae(pool_classifiers,
                                                 X_DSEL[:n_samples],
                                                 y_DSEL[:n_samples],
                                                 X_test[:n_t],
                                                 y_test[:n_t],
                                                 knn_type='knn')

        print("sklearn_knorae score = {}, time = {}".format(score_sklearn,
                                                            time_sklearn))

        score_faiss, time_faiss = run_knorae(pool_classifiers,
                                             X_DSEL[:n_samples],
                                             y_DSEL[:n_samples],
                                             X_test[:n_t],
                                             y_test[:n_t],
                                             knn_type='faiss')

        print("faiss_knorae score = {}, time = {}".format(score_faiss,
                                                          time_faiss))
