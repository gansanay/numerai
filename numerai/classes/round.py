# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import os
from sklearn import cross_validation as CV
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.metrics import roc_auc_score as AUC
from pkg_resources import resource_filename
from time import ctime
import pkgutil


class Round(object):

    def __init__(self, round_number):
        self.round_number = round_number
        self.train_file_name = 'r' + str(round_number) + '_numerai_training_data.csv'
        self.test_file_name = 'r' + str(round_number) + '_numerai_tournament_data.csv'
        self.sorted_file_name = 'r' + str(round_number) + '_numerai_sorted_training_data.csv'

        if not os.path.exists(resource_filename('numerai.data', self.train_file_name)):
            raise IOError('File {} not found.'.format(self.train_file_name))

        if not os.path.exists(resource_filename('numerai.data', self.test_file_name)):
            raise IOError('File {} not found.'.format(self.test_file_name))

    def has_sorted_training_set(self):
        try:
            pkgutil.get_data('numerai.data', 'r' + str(self.round_number) + '_numerai_sorted_training_data.csv')
            return True
        except IOError:
            return False

    def training_set(self):
        return pd.read_csv(resource_filename('numerai.data', self.train_file_name))

    def test_set(self):
        return pd.read_csv(resource_filename('numerai.data', self.test_file_name))

    def sorted_training_set(self):
        return pd.read_csv(resource_filename('numerai.data', self.sorted_file_name))

    def sort_training_set(self, classifier='RF'):

        print "loading..."

        train = pd.read_csv(resource_filename('numerai.data', self.train_file_name))
        test = pd.read_csv(resource_filename('numerai.data', self.test_file_name))

        test.drop('t_id', axis=1, inplace=True)
        test['target'] = 0  # dummy for preserving column order when concatenating

        train['is_test'] = 0
        test['is_test'] = 1

        orig_train = train.copy()
        assert (np.all(orig_train.columns == test.columns))

        train = pd.concat((orig_train, test))
        train.reset_index(inplace=True, drop=True)

        x = train.drop(['is_test', 'target'], axis=1)
        y = train.is_test

        print "cross-validating..."

        n_estimators = 100
        if classifier == 'RF':
            clf = RF(bootstrap=True,
                     min_samples_leaf=3,
                     n_estimators=n_estimators,
                     max_features=20,
                     criterion='gini',
                     min_samples_split=20,
                     max_depth=None,
                     n_jobs=6)
        else:
            clf = LR(n_jobs=6)

        predictions = np.zeros(y.shape)

        cv = CV.StratifiedKFold(y, n_folds=5, shuffle=True, random_state=5678)

        for f, (train_i, test_i) in enumerate(cv):
            print "# fold {}, {}".format(f + 1, ctime())

            x_train = x.iloc[train_i]
            x_test = x.iloc[test_i]
            y_train = y.iloc[train_i]
            y_test = y.iloc[test_i]

            clf.fit(x_train, y_train)

            p = clf.predict_proba(x_test)[:, 1]

            auc = AUC(y_test, p)
            print "# AUC: {:.2%}\n".format(auc)

            predictions[test_i] = p

        train['p'] = predictions

        i = predictions.argsort()
        train_sorted = train.iloc[i]

        train_sorted = train_sorted.loc[train_sorted.is_test == 0]
        assert (train_sorted.target.sum() == orig_train.target.sum())

        train_sorted.drop('is_test', axis=1, inplace=True)
        train_sorted.to_csv(resource_filename('numerai.data', self.sorted_file_name), index=False)


if __name__ == '__main__':

    r44 = Round(44)
    print r44.has_sorted_training_set()
    r44.sort_training_set(classifier='LR')