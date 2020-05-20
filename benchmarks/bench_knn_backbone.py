import time
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from deslib.util.faiss_knn_wrapper import FaissKNNClassifier

n_samples = [1000, 10000, 100000, 1000000, 10000000]
rng = 42

faiss_brute = FaissKNNClassifier(n_neighbors=7,
                                 algorithm='brute')
faiss_voronoi = FaissKNNClassifier(n_neighbors=7,
                                   algorithm='voronoi')
faiss_hierarchical = FaissKNNClassifier(n_neighbors=7,
                                        algorithm='hierarchical')

all_knns = [faiss_brute, faiss_voronoi, faiss_hierarchical]
names = ['faiss_brute', 'faiss_voronoi', 'faiss_hierarchical']

list_fitting_time = []
list_search_time = []

for n in n_samples:

    print("Number of samples: {}" .format(n))
    X, y = make_classification(n_samples=n,
                               n_features=20,
                               random_state=rng)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=rng)
    temp_fitting_time = []
    temp_search_time = []
    for name, knn in zip(names, all_knns):
        start = time.clock()
        knn.fit(X_train, y_train)
        fitting_time = time.clock() - start
        print("{} fitting time: {}" .format(name, fitting_time))

        start = time.clock()
        neighbors, dists = knn.kneighbors(X_test)
        search_time = time.clock() - start
        print("{} neighborhood search time: {}" .format(name, search_time))

        temp_fitting_time.append(fitting_time)
        temp_search_time.append(search_time)

    list_fitting_time.append(temp_fitting_time)
    list_search_time.append(temp_search_time)

plt.plot(n_samples, list_search_time)
plt.legend(names)
plt.xlabel("Number of samples")
plt.ylabel("K neighbors search time")
plt.savefig('knn_backbone_benchmark.png')
