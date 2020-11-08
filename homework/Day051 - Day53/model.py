import joblib
import glob
import pandas as pd
import numpy as np
from feature_engineering import read_data
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import RandomizedSearchCV
from sklearn import ensemble
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier


def run_random_search_cv(train_feature, train_label, clf, param_grid, print_best_params=False):
    # Split dataframe into training and validation dataset
    X_train, X_valid, y_train, y_valid = train_test_split(
        train_feature, train_label, test_size=0.2, stratify=train_label, random_state=914)
    # Start random search
    rs_clf = RandomizedSearchCV(
        estimator=clf, param_distributions=param_grid, n_iter=20,
        n_jobs=-1, verbose=0, cv=10, scoring='roc_auc', refit=True, random_state=914)
    rs_clf.fit(X_train, y_train)
    best_score = rs_clf.best_score_
    print("Mean cross-validated score: {:.4f}".format(best_score))
    if print_best_params:
        best_params = rs_clf.best_params_
        print("Best params: ")
        for param_name in sorted(best_params.keys()):
            print('%s: %r' % (param_name, best_params[param_name]))
    # Refit classifier using best parameters from cv
    clf = rs_clf.best_estimator_
    clf.fit(X_train, y_train)
    # Predict probability and calculate auc score
    prob = clf.predict_proba(X_valid)[:, 1]
    auc = metrics.roc_auc_score(y_valid, prob)
    print("Score after refit: {:.4f}\n".format(auc))
    return clf


def blending(train_feature, train_label):
    names = [
        "randomforest",
        "xgboost", 
        "catboost", 
        "lgmboost"
        ]
    classifiers = [
        ensemble.RandomForestClassifier(),
        XGBClassifier(), 
        CatBoostClassifier(verbose=0), 
        LGBMClassifier(silent=True)
    ]
    parameters = [
         # RandomForest
        {
            'bootstrap': [True, False],
            'max_depth': [int(x) for x in range(10, 50)],
            'max_features': ['auto', 'sqrt'],
            'min_samples_leaf': [int(x) for x in range(1, 5)],
            'min_samples_split': [int(x) for x in range(2, 10)],
            'n_estimators': [int(x) for x in range(100, 500, 50)]
         }, 
         # XGBoost
        {
            'silent': [False],
            'max_depth': [6, 10, 15, 20],
            'learning_rate': [0.001, 0.01, 0.1, 0.2, 0,3],
            'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'colsample_bylevel': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'min_child_weight': [0.5, 1.0, 3.0, 5.0, 7.0, 10.0],
            'gamma': [0, 0.25, 0.5, 1.0],
            'reg_lambda': [0.1, 1.0, 5.0, 10.0, 50.0, 100.0],
            'n_estimators': [100]
        }, 
        # CatBoost
        {
            'max_depth': [int(x) for x in range(4, 11)], 
            'iterations': [int(x) for x in range(10, 100)]
        }, 
        # LGMBoost
        {
            'n_estimators': [int(x) for x in range(100, 500, 50)]
        }
    ]
    prob_list = []
    X_train, X_valid, y_train, y_valid = train_test_split(
        train_feature, train_label, test_size=0.5, stratify=train_label, random_state=914)
    for name, classifier, param_dist in zip(names, classifiers, parameters):
        print(name)
        print("-"*50)
        clf = run_random_search_cv(train_feature, train_label, classifier, param_dist)
        joblib.dump(clf, f"./model/{name}.bin", compress=5)
        prob = clf.predict_proba(X_valid)[:, 1]
        prob_list.append(prob)
    blending_prob = np.mean(prob_list, axis=0)
    auc = metrics.roc_auc_score(y_valid, blending_prob)
    print("Score after blending: {:.4f}\n".format(auc))


def submit_blending_prediction():
    train_feature, test_feature, train_label = read_data()
    files = glob.glob("./model/*.bin")
    model_predictions = []
    for f in files:
        clf = joblib.load(f)
        prob = clf.predict_proba(test_feature)[:, 1]
        model_predictions.append(prob)
    blending_prob = np.mean(model_predictions, axis=0)
    submit = pd.read_csv("./data/sample_submission.csv")
    submit['poi'] = blending_prob
    submit.to_csv('Submission_20201108.csv', index=False)


def main1():
    train_feature, test_feature, train_label = read_data()
    clf = XGBClassifier()
    blending(train_feature, train_label)


def main2():
    submit_blending_prediction()


if __name__ == "__main__":
    main2()