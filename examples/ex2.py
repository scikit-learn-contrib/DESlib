from deslib.dcs.a_priori import APriori
from deslib.dcs.mcb import MCB
# Example of a dcs techniques
from deslib.dcs.ola import OLA
from deslib.des.des_p import DESP
from deslib.des.knora_u import KNORAU
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    # Split the data into training and DSEL for DS techniques
    X_train, X_dsel, y_train, y_dsel = train_test_split(X_train, y_train, test_size=0.5)
    # Considering a pool composed of 10 base classifiers

    # Training a random forest to be used as the pool of classifiers
    pool_classifiers = RandomForestClassifier(n_estimators=10)
    pool_classifiers.fit(X_train, y_train)

    # Initializa a DS technique. Here we specify the size of the region of competence (5 neighbors)
    knorau = KNORAU(pool_classifiers)
    kne = KNORAE(pool_classifiers, k=5)
    desp = DESP(pool_classifiers, k=5)
    ola = OLA(pool_classifiers, k=5)
    mcb = MCB(pool_classifiers, k=5)
    apriori = APriori(pool_classifiers, k=5)

    # Fit the DS techniques
    knorau.fit(X_dsel, y_dsel)
    kne.fit(X_dsel, y_dsel)
    desp.fit(X_dsel, y_dsel)

    ola.fit(X_dsel, y_dsel)
    mcb.fit(X_dsel, y_dsel)
    apriori.fit(X_dsel, y_dsel)

    # Calculate classification accuracy of each technique
    print('Classification accuracy RF: ', pool_classifiers.score(X_test, y_test))
    print('Classification accuracy KNORAU: ', knorau.score(X_test, y_test))
    print('Classification accuracy KNORA-Eliminate: ', kne.score(X_test, y_test))
    print('Classification accuracy DESP: ', desp.score(X_test, y_test))
    print('Classification accuracy OLA: ', ola.score(X_test, y_test))
    print('Classification accuracy A priori: ', apriori.score(X_test, y_test))





