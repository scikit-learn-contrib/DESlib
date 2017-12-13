# Probabilistic DS. To apply such methods, the base classifiers must have a good estimation of probabilities.
# Otherwise, these techniques presents poor estimates.

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from pythonds.des.probabilistic import DESKL
# Example of probabilistic des techniques
from pythonds.des.probabilistic import RRC

if __name__ == "__main__":

    # Generate a classification dataset
    data = load_breast_cancer()
    X = data.data
    y = data.target
    # split the data into training and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    # Split the data into training and DSEL for DS techniques
    X_train, X_dsel, y_train, y_dsel = train_test_split(X_train, y_train, test_size=0.5)

    # model = CalibratedClassifierCV(Perceptron(), cv=5)
    # Train a pool of 10 classifiers
    pool_classifiers = BaggingClassifier(base_estimator=LogisticRegression(), n_estimators=10)
    pool_classifiers.fit(X_train, y_train)

    # Initialize the DS techniques
    rrc = RRC(pool_classifiers)
    deskl = DESKL(pool_classifiers)

    rrc.fit(X_dsel, y_dsel)
    deskl.fit(X_dsel, y_dsel)

    # Calculate classification accuracy of each technique
    print('Classification accuracy Bagging: ', pool_classifiers.score(X_test, y_test))
    print('Classification accuracy Randomized Reference Classifier: ', rrc.score(X_test, y_test))
    print('Classification accuracy Kullback-Leibler: ', deskl.score(X_test, y_test))








