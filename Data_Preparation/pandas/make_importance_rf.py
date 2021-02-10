#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot as plt


def main():
    args = {
        'n_samples': 1000,
        'n_features': 100,
        'n_informative': 5,
        'n_redundant': 0,
        'n_repeated': 0,
        'n_classes': 2,
        'random_state': 42,
        'shuffle': False,
    }
    X, y = make_classification(**args)

    clf = RandomForestClassifier(n_estimators=100,
                                 random_state=42)

    # 全データを使って学習する
    clf.fit(X, y)

    # ランダムフォレストから特徴量の重要度を取り出す
    importances = clf.feature_importances_
    importance_std = np.std([tree.feature_importances_
                             for tree in clf.estimators_],
                            axis=0)
    indices = np.argsort(importances)[::-1]

    # TOP 20 の特徴量を出力する
    rank_n = min(X.shape[1], 20)
    print('Feature importance ranking (TOP {rank_n})'.format(rank_n=rank_n))
    for i in range(rank_n):
        params = {
            'rank': i + 1,
            'idx': indices[i],
            'importance': importances[indices[i]]
        }
        print('{rank}. feature {idx:02d}: {importance}'.format(**params))

    # TOP 20 の特徴量の重要度を可視化する
    plt.figure(figsize=(8, 32))
    plt.barh(range(rank_n),
             importances[indices][:rank_n],
             color='g',
             ecolor='r',
             yerr=importance_std[indices][:rank_n],
             align='center')
    plt.yticks(range(rank_n), indices[:rank_n])
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.grid()
    plt.show()


if __name__ == '__main__':
    main()
