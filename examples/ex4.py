# Example applying the Dynamic Frienemy Pruning algorithm.
from pythonds.dcs.a_priori import APriori
# Example of dcs techniques
from pythonds.dcs.ola import OLA
from pythonds.des.des_p import DESP
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split

# Example of des techniques
from pythonds.des.knora_e import KNORAE

data = load_breast_cancer()
X = data.data
y = data.target

# split the data into training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

# Split the data into training and DSEL for DS techniques
X_train, X_dsel, y_train, y_dsel = train_test_split(X_train, y_train, test_size=0.5)

# Calibrating Perceptrons to estimate probabilities
model = CalibratedClassifierCV(Perceptron())
pool_classifiers = BaggingClassifier(n_estimators=10)
pool_classifiers.fit(X_train, y_train)

# DS techniques without DFP
knorae = KNORAE(pool_classifiers)
ola = OLA(pool_classifiers)
apriori = APriori(pool_classifiers)
desp = DESP(pool_classifiers)

# DS techniques using DFP
fire_knorae = KNORAE(pool_classifiers, DFP=True)
fire_ola = OLA(pool_classifiers, DFP=True)
fire_apriori = APriori(pool_classifiers, DFP=True)
fire_desp = DESP(pool_classifiers, DFP=True)

knorae.fit(X_dsel, y_dsel)
ola.fit(X_dsel, y_dsel)
apriori.fit(X_dsel, y_dsel)
desp.fit(X_dsel, y_dsel)

fire_knorae.fit(X_dsel, y_dsel)
fire_ola.fit(X_dsel, y_dsel)
fire_apriori.fit(X_dsel, y_dsel)
fire_desp.fit(X_dsel, y_dsel)

print('Classification accuracy of KNORAE: ', knorae.score(X_test, y_test))
print('Classification accuracy of Fire-KNORAE: ', fire_knorae.score(X_test, y_test))

print('Classification accuracy of OLA: ', ola.score(X_test, y_test))
print('Classification accuracy of Fire-OLA: ', fire_ola.score(X_test, y_test))

print('Classification accuracy of A priori: ', apriori.score(X_test, y_test))
print('Classification accuracy of Fire-A priori: ', fire_apriori.score(X_test, y_test))

print('Classification accuracy of des-P: ', desp.score(X_test, y_test))
print('Classification accuracy of Fire-des-P: ', fire_desp.score(X_test, y_test))