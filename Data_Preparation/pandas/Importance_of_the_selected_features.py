from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedKFold


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

    # Recursive Feature Elimination
    rfe = RFE(estimator=clf,
              # 有効そうな 5 つの特徴量を取り出す
              n_features_to_select=5,
              verbose=1)

    # 特徴量の選択と評価のためにデータを分割する
    # 計算量が許すのであれば k-Fold した方が bias は小さくなるはず
    X_train, X_eval, y_train, y_eval = train_test_split(X, y,
                                                        shuffle=True,
                                                        random_state=42)

    # RFE を学習する
    rfe.fit(X_eval, y_eval)

    # RFE による特徴量の評価 (ランキング)
    print('Feature ranking by RFF:', rfe.ranking_)

    # RFE で選択された特徴量だけを取り出す
    X_train_selected = X_train[:, rfe.support_]

    # Stratified 5-Fold CV で OOF Prediction (probability) を作る
    skf = StratifiedKFold(n_splits=5,
                          shuffle=True,
                          random_state=42)
    y_pred = cross_val_predict(clf, X_train_selected, y_train,
                               cv=skf,
                               method='predict_proba')

    # AUC のスコアを確認する
    metric = roc_auc_score(y_train, y_pred[:, 1])
    print('RFE selected features AUC:', metric)


if __name__ == '__main__':
    main()
