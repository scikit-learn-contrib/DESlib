import numpy as np
import faiss
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import time
from sklearn.model_selection import train_test_split
import threading
import os
import warnings

def sk_knn(Xtrain, Y, k, Xtest):
    start = time.clock()
    s_knn = KNeighborsClassifier(k, n_jobs=4) #Half of current cores
    s_knn.fit(Xtrain, Y)
    s_knn.predict(Xtest)
    print("sklearn_knn run_time: {}".format(time.clock() - start))

def faiss_knn(Xtrain, Y, k, Xtest):
    start = time.clock()
    index = faiss.IndexFlatL2(Xtrain.shape[1])
    index.add(np.ascontiguousarray(Xtrain).astype(np.float32))
    index.search(Xtest.astype(np.float32), k)
    print("faiss_knn run_time: {}".format(time.clock() - start))


if __name__ == "__main__":

    if not os.path.exists("../../HIGGS.csv"):
        warnings.warn("Please download the HIGGS dataset from https://archive.ics.uci.edu/ml/datasets/HIGGS")

    df = pd.read_csv('../../HIGGS.csv', header=None)
    data = df.values
    X = data[:, 1:]
    Y = data[:, 0]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33)
    num_samples_list = [1000000]
    num_of_k_list = [1, 2, 5, 7, 10]
    num_of_test_inputs = [100, 1000]

    for nsamples in num_samples_list:
        for n_k in num_of_k_list:
            for n_t in num_of_test_inputs:
                print("running experiment: num_of_train_samples: {}, num_of_k: {}, num_of_tests: {}".format(
                    nsamples,
                    n_k,
                    n_t))
                faiss_knn(X_train[:nsamples], Y_train[:nsamples], n_k, X_test[:n_t])
                t = threading.Thread(target=sk_knn, args=(X_train[:nsamples], Y_train[:nsamples], n_k, X_test[:n_t]))
                t.start()
                t.join(timeout=600)
                if t.is_alive():
                    print("sklearn_knn, num_of_train_samples: {}, num_of_k: {}, num_of_tests: {}, run_time: {}".format(
                        nsamples,
                        n_k,
                        n_t,
                        "timeout after 60s"))
