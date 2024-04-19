import numpy as np
import pytest
import math
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import precision_recall_fscore_support as prf
from sklearn.calibration import CalibratedClassifierCV as CC
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import GaussianNB as GNB
from deslib.base import BaseDS
from deslib.multi_datasets import MultiDatasets
# Static techniques
from deslib.static.oracle import Oracle
from deslib.static.stacked import StackedClassifier
# DCS techniques
from deslib.dcs.a_posteriori import APosteriori
from deslib.dcs.a_priori import APriori
from deslib.dcs.lca import LCA
from deslib.dcs.mcb import MCB
from deslib.dcs.mla import MLA
from deslib.dcs.ola import OLA
from deslib.dcs.rank import Rank
from deslib.des import DESKL
# DES techniques
from deslib.des.des_knn import DESKNN
from deslib.des.des_p import DESP
from deslib.des.knop import KNOP
from deslib.des.knora_e import KNORAE
from deslib.des.knora_u import KNORAU
from deslib.des.meta_des import METADES


# ----- Integration tests -----

def setup_classifiers():
    rng = np.random.RandomState(123456)
    rng2 = np.random.RandomState(654321)

    # Generate a classification dataset
    X, y = make_classification(n_classes=2, n_samples=1000, weights=[0.2, 0.8],
                               random_state=rng)
    X2, y2 = make_classification(n_classes=2, n_samples=1000, weights=[0.3, 0.7],
                               random_state=rng2)
    # split the data into training and test data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=rng)
    X_train2, X_test2, y_train2, y_test2 = train_test_split(
        X2, y2, test_size=0.5, random_state=rng2)

    # Scale the variables to have 0 mean and unit variance
    scalar = StandardScaler()
    X_train = scalar.fit_transform(X_train)
    X_test = scalar.transform(X_test)
    scalar2 = StandardScaler()
    X_train2 = scalar2.fit_transform(X_train2)
    X_test2 = scalar2.transform(X_test2)

    # Split the data into training and DSEL for DS techniques
    X_train, X_dsel, y_train, y_dsel = train_test_split(
        X_train, y_train, test_size=0.5, random_state=rng)
    X_train2, X_dsel2, y_train2, y_dsel2 = train_test_split(
        X_train2, y_train2, test_size=0.5, random_state=rng2)
    
    gnb1 = GNB()
    gnb2 = GNB()
    gnb1.fit(X_train, y_train)
    gnb2.fit(X_train2, y_train2)
    pool_classifiers = np.asarray([[gnb1, gnb1], [gnb2, gnb2]])
    return (X_dsel, y_dsel, X_test, y_test,
            X_dsel2, y_dsel2, X_test2, y_test2, pool_classifiers)

@pytest.mark.parametrize('params',[
    [Oracle(),0.962],
    [StackedClassifier(),0.85],
    [KNORAU(),0.764],
    [KNORAE(),0.772],
    [DESP(),0.666],
    [OLA(),0.830],
    [LCA(),0.814],
    [MLA(),0.810],
    [MCB(random_state=0),0.806],
    [APriori(random_state=0),0.796],
    [Rank(),0.824],
    [APosteriori(random_state=0),0.782],
    [METADES(),0.690],
    [KNOP(),0.792],
    [DESKL(),0.680]
])
def test(params):
    [X_dsel, y_dsel, X_test, y_test,
     X_dsel2, y_dsel2, X_test2, y_test2, pool_classifiers] = setup_classifiers()
    technique = MultiDatasets(params[0], pool_classifiers)
    technique.fit(np.asarray([X_dsel, X_dsel2]), np.asarray([y_dsel, y_dsel2]))
    pred = technique.predict(np.asarray([X_test, X_test2]), y_test);
    precision = prf(y_test, pred, average='micro')[0]
    assert math.isclose(precision, params[1]);

