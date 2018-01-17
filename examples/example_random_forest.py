"""
In this example we use a pool of classifiers generated using the Random Forest method rather than Bagging. We also
show how to change the size of the region of competence, used to estimate the local competence of the base classifiers.

This demonstrates that the library accepts any kind of base classifiers as long as they implement the predict and
predict proba functions. Moreover, any ensemble generation method such as Boosting or Rotation Trees can be used
to generate a pool containing diverse base classifiers. We also included the performance of the RandomForest classifier
as a baseline comparison.
"""

# Example of a dcs techniques
from deslib.dcs.ola import OLA
from deslib.dcs.mcb import MCB
from deslib.des.des_p import DESP
from deslib.des.knora_u import KNORAU
from deslib.des.meta_des import METADES
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Example of a des techniques
from deslib.des.knora_e import KNORAE

if __name__ == "__main__":

    # Generate a classification dataset
    data = load_breast_cancer()
    X = data.data
    y = data.target
    # split the data into training and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    RF = RandomForestClassifier()
    RF.fit(X_train, y_train)

    X_train, X_dsel, y_train, y_dsel = train_test_split(X, y, test_size=0.50)

    # Training a random forest to be used as the pool of classifiers. We set the maximum depth of the tree so that it
    # can estimate probabilities
    pool_classifiers = RandomForestClassifier(n_estimators=10, max_depth=5)
    pool_classifiers.fit(X_train, y_train)

    # Initialize a DS technique. Here we specify the size of the region of competence (5 neighbors)
    knorau = KNORAU(pool_classifiers)
    kne = KNORAE(pool_classifiers, k=5)
    desp = DESP(pool_classifiers, k=5)
    ola = OLA(pool_classifiers, k=5)
    mcb = MCB(pool_classifiers, k=5)
    meta = METADES(pool_classifiers, k=5)

    # Fit the DS techniques
    knorau.fit(X_dsel, y_dsel)
    kne.fit(X_dsel, y_dsel)
    desp.fit(X_dsel, y_dsel)
    meta.fit(X_dsel, y_dsel)
    ola.fit(X_dsel, y_dsel)
    mcb.fit(X_dsel, y_dsel)

    # Calculate classification accuracy of each technique
    print('Classification accuracy RF: ', RF.score(X_test, y_test))
    print('Evaluating DS techniques:')
    print('Classification accuracy KNORAU: ', knorau.score(X_test, y_test))
    print('Classification accuracy KNORA-Eliminate: ', kne.score(X_test, y_test))
    print('Classification accuracy DESP: ', desp.score(X_test, y_test))
    print('Classification accuracy OLA: ', ola.score(X_test, y_test))
    print('Classification accuracy MCB: ', mcb.score(X_test, y_test))
    print('Classification accuracy META-DES: ', meta.score(X_test, y_test))





