.. _introduction:

==============
Introduction
==============

Dynamic Selection (DS) refers to techniques in which the base classifiers are selected
on the fly, according to each new sample to be classified. Only the most competent, or an ensemble containing the most competent classifiers is selected to predict
the label of a specific test sample. The rationale for such techniques is that not every classifier in
the pool is an expert in classifying all unknown samples; rather, each base classifier is an expert in
a different local region of the feature space.

DS is one of the most promising MCS approaches due to the fact that
more and more works are reporting the superior performance of such techniques over static combination methods. Such techniques
have achieved better classification performance especially when dealing with small-sized and imbalanced datasets.


References:
-----------
.. [1] : R. M. O. Cruz, R. Sabourin, and G. D. Cavalcanti, “Dynamic classifier selection: Recent advances and perspectives,” Information Fusion, vol. 41, pp. 195 – 216, 2018.

.. [2] : A. S. Britto, R. Sabourin, L. E. S. de Oliveira, Dynamic selection of classifiers - A comprehensive review, Pattern Recognition 47 (11) (2014) 3665–3680.
