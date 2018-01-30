"""
In this example we show that the framework can also be used using different classifier models in the pool
of classifiers. Such pool of classifiers are called Heterogeneous.

Here we consider a pool of classifiers composed of a Gaussian Naive Bayes, Perceptron, k-NN, Decision tree
Linear SVM and Gaussian SVM
"""

# Importing dynamic selection techniques:
from deslib.dcs.a_posteriori import APosteriori
from deslib.dcs.mcb import MCB
from deslib.dcs.lca import LCA
from deslib.des.probabilistic import RRC
from deslib.des.knop import KNOP
from deslib.des.knora_e import KNORAE

# Base classifier models:
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.calibration import CalibratedClassifierCV

# Importing dataset
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler


if __name__ == "__main__":
    # Generate a classification dataset
    data = load_breast_cancer()
    X = data.data
    y = data.target

    # split the data into training and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    # Scale the variables to have 0 mean and unit variance
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Split the data into training and DSEL for DS techniques
    X_train, X_dsel, y_train, y_dsel = train_test_split(X, y, test_size=0.5)

    model_perceptron = CalibratedClassifierCV(Perceptron(max_iter=100)).fit(X_train, y_train)
    model_linear_svm = CalibratedClassifierCV(LinearSVC()).fit(X_train, y_train)
    model_svc = SVC(probability=True).fit(X_train, y_train)
    model_bayes = GaussianNB().fit(X_train, y_train)
    model_tree = DecisionTreeClassifier().fit(X_train, y_train)
    model_knn = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
    pool_classifiers = [model_perceptron, model_linear_svm, model_svc, model_bayes,  model_tree, model_knn]

    # Initializing the DS techniques
    knop = KNOP(pool_classifiers)
    rrc = RRC(pool_classifiers)
    lca = LCA(pool_classifiers)
    mcb = MCB(pool_classifiers)
    aposteriori = APosteriori(pool_classifiers)

    # Fitting the techniques
    knop.fit(X_dsel, y_dsel)
    rrc.fit(X_dsel, y_dsel)
    lca.fit(X_dsel, y_dsel)
    mcb.fit(X_dsel, y_dsel)
    aposteriori.fit(X_dsel, y_dsel)

    # Calculate classification accuracy of each technique
    print('Evaluating DS techniques:')
    print('Classification accuracy KNOP: ', knop.score(X_test, y_test))
    print('Classification accuracy RRC: ', rrc.score(X_test, y_test))
    print('Classification accuracy LCA: ', lca.score(X_test, y_test))
    print('Classification accuracy A posteriori: ', aposteriori.score(X_test, y_test))





