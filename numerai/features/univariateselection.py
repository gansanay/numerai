# -*- coding: utf-8 -*-

import pandas as pd
from scipy.stats import pearsonr
from scipy.stats import ks_2samp
from numerai import Round


def pearson(X, y):
    r = []
    p = []
    for c in X.columns:
        r_, p_ = pearsonr(X[c], y)
        r.append(r_)
        p.append(p_)
    dfr = pd.DataFrame(index=range(1, 1+len(X.columns)))
    dfr['pearson'] = r
    dfr['pearson_p'] = p
    return dfr

def kolmogorov_smirnov(x_train, x_test):
    r = []
    p = []
    for c in x_train.columns:
        r_, p_ = ks_2samp(x_train[c], x_test[c])
        r.append(r_)
        p.append(p_)
    dfks = pd.DataFrame(index=range(1, 1 + len(x_train.columns)))
    dfks['KS'] = r
    dfks['KS_p'] = p
    return dfks

if __name__ == '__main__':

    r44 = Round(44)
    train = r44.training_set()
    print pearson(train.drop('target', axis=1), train.target).head()

