from deslib.dcs.a_priori import APriori
from deslib.dcs.mcb import MCB
# Example of a dcs techniques
from deslib.dcs.ola import OLA
from deslib.des.des_p import DESP
from deslib.des.knora_u import KNORAU
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import load_breast_cancer
# Base classifier models
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Example of a des techniques
from deslib.des.knora_e import KNORAE

if __name__ == "__main__":
    # Generate a classification dataset
    # X, y = make_classification(n_classes=2, n_samples=2000, n_features=20)
    data = load_breast_cancer()
    X = data.data
    y = data.target
    # split the data into training and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    # Split the data into training and DSEL for DS techniques
    X_train2, X_dsel, y_train2, y_dsel = train_test_split(X, y, test_size=0.5)

    model_perceptron = CalibratedClassifierCV(Perceptron(), cv=3, method='sigmoid').fit(X_train, y_train)
    model_linear_svm = CalibratedClassifierCV(LinearSVC(), cv=3, method='sigmoid').fit(X_train, y_train)
    model_svc = SVC(probability=True).fit(X_train, y_train)
    model_bayes = GaussianNB().fit(X_train, y_train)
    model_bayes2 = MultinomialNB().fit(X_train, y_train)
    model_tree = DecisionTreeClassifier().fit(X_train, y_train)
    model_knn = KNeighborsClassifier(n_neighbors=1).fit(X_train, y_train)
    pool_classifiers = [model_perceptron, model_linear_svm, model_svc, model_bayes, model_bayes2, model_tree, model_knn]

    # Initializa a DS technique
    knorau = KNORAU(pool_classifiers, DFP=True)
    kne = KNORAE(pool_classifiers, DFP=True)
    desp = DESP(pool_classifiers, DFP=True)
    ola = OLA(pool_classifiers, DFP=True)
    mcb = MCB(pool_classifiers, DFP=True)
    apriori = APriori(pool_classifiers, DFP=True)

    # Fit the des techniques
    knorau.fit(X_dsel, y_dsel)
    kne.fit(X_dsel, y_dsel)
    desp.fit(X_dsel, y_dsel)

    # Fit the dcs techniques
    ola.fit(X_dsel, y_dsel)
    mcb.fit(X_dsel, y_dsel)
    apriori.fit(X_dsel, y_dsel)

    # Calculate classification accuracy of each technique
    print('Classification accuracy KNORAU: ', knorau.score(X_test, y_test))
    print('Classification accuracy KNORA-Eliminate: ', kne.score(X_test, y_test))
    print('Classification accuracy DESP: ', desp.score(X_test, y_test))
    print('Classification accuracy OLA: ', ola.score(X_test, y_test))
    print('Classification accuracy A priori: ', apriori.score(X_test, y_test))





