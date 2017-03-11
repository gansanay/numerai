# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from sklearn.learning_curve import learning_curve
from sklearn import decomposition
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5), filename='learningcurve'):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, scoring='log_loss', cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.savefig(filename+'.png')


def plot_PCA(train, test, label, filename, show=False):

    pca_train = train.copy()
    pca_test = test.copy()
    pca_test['target'] = 0  # dummy for preserving column order when concatenating

    pca_train[label] = 0
    pca_test[label] = 1

    pca_mix = pd.concat((pca_train, pca_test))

    pca_X = pca_mix.drop([label, 'target'], axis=1)
    pca_y = pca_mix[label]

    pca = decomposition.PCA(n_components=2)
    pca.fit(pca_X)

    dfx = pd.DataFrame(pca.transform(pca_X))
    dfy = pd.DataFrame(pca_y.values)
    df_ = pd.concat([dfx, dfy], axis=1)
    df_.columns = ['x0', 'x1', 'y']

    fig, axScatter = plt.subplots(figsize=(15, 15))

    # the scatter plot:
    # axScatter.scatter(X[:,0], X[:,1], c=y, lw=0, s=1, alpha=0.2)
    axScatter.scatter(df_[df_.y == 0]['x0'], df_[df_.y == 0]['x1'], c='b', lw=0, s=1, alpha=0.5)
    axScatter.scatter(df_[df_.y == 1]['x0'], df_[df_.y == 1]['x1'], c='g', lw=0, s=1, alpha=0.5)
    axScatter.set_aspect(1.)
    axScatter.set_aspect(1.)

    # create new axes on the right and on the top of the current axes
    # The first argument of the new_vertical(new_horizontal) method is
    # the height (width) of the axes to be created in inches.
    divider = make_axes_locatable(axScatter)
    axHistx = divider.append_axes("top", 1.8, pad=0.1, sharex=axScatter)
    axHisty = divider.append_axes("right", 1.8, pad=0.1, sharey=axScatter)

    # make some labels invisible
    plt.setp(axHistx.get_xticklabels() + axHisty.get_yticklabels(),
             visible=False)

    # now determine nice limits by hand:
    binwidth = 0.05
    xymax = np.max([np.max(np.fabs(df_['x0'].values)), np.max(np.fabs(df_['x1'].values))])
    lim = (int(xymax / binwidth) + 1) * binwidth

    bins = np.arange(-lim, lim + binwidth, binwidth)
    axHistx.hist([df_[df_.y == 0]['x0'].values, df_[df_.y == 1]['x0'].values], bins=bins, normed=True)
    axHisty.hist([df_[df_.y == 0]['x1'].values, df_[df_.y == 1]['x1'].values], bins=bins, orientation='horizontal',
                 normed=True)

    # the xaxis of axHistx and yaxis of axHisty are shared with axScatter,
    # thus there is no need to manually adjust the xlim and ylim of these
    # axis.

    # axHistx.axis["bottom"].major_ticklabels.set_visible(False)
    for tl in axHistx.get_xticklabels():
        tl.set_visible(False)
    axHistx.set_yticks([0, 0.5, 1])

    # axHisty.axis["left"].major_ticklabels.set_visible(False)
    for tl in axHisty.get_yticklabels():
        tl.set_visible(False)
    axHisty.set_xticks([0, 0.5, 1])

    plt.draw()
    if show:
        plt.show()
    fig.savefig(filename + '.png')