import gzip
import os
import shutil
import threading
import time
import urllib.request

import pandas as pd
from sklearn.model_selection import train_test_split

from deslib.des.knora_e import KNORAE


def sk_KNORAE_knn(XTrain, YTrain, k, XTest, YTest):
    start = time.clock()
    knorae_sk = KNORAE(k=k, knn_classifier='knn')
    knorae_sk.fit(XTrain, YTrain)
    score = knorae_sk.score(XTest, YTest)
    print("sklearn_knn_knorae run_time: {}".format(time.clock() - start))
    print("sklearn_knn_knorae score: {}".format(score))


def faiss_KNORAE_knn(XTrain, YTrain, k, XTest, YTest):
    start = time.clock()
    knorae_sk = KNORAE(k=k, knn_classifier='faiss')
    knorae_sk.fit(XTrain, YTrain)
    score = knorae_sk.score(XTest, YTest)
    print("faiss_knn_knorae run_time: {}".format(time.clock() - start))
    print("faiss_knn_knorae score: {}".format(score))


if __name__ == "__main__":
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

    df = pd.read_csv('../../HIGGS.csv', header=None)
    data = df.values
    X = data[:, 1:]
    Y = data[:, 0]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33)
    num_samples_list = [1000000]
    num_of_k_list = [2, 5, 7, 10]
    num_of_test_inputs = [100, 1000, 10000]

    for nsamples in num_samples_list:
        for n_k in num_of_k_list:
            for n_t in num_of_test_inputs:
                print("running experiment: num_of_train_samples: {}, "
                      "num_of_k: {}, num_of_tests: {}".format(nsamples, n_k,
                                                              n_t))
                faiss_KNORAE_knn(X_train[:nsamples], Y_train[:nsamples], n_k,
                                 X_test[:n_t], Y_test[:n_t])
                t = threading.Thread(target=sk_KNORAE_knn, args=(
                X_train[:nsamples], Y_train[:nsamples], n_k, X_test[:n_t],
                Y_test[:n_t]))

                t.start()
                t.join(timeout=600)
                if t.is_alive():
                    print(
                        "sklearn_knn, num_of_train_samples: {}, num_of_k: {}, "
                        "num_of_tests: {}, run_time: timeout".format(nsamples,
                                                                     n_k,
                                                                     n_t))
