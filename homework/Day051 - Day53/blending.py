import joblib
import glob
import warnings
import pandas as pd
import numpy as np
from random import choice
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn import linear_model
from sklearn import naive_bayes
from sklearn import ensemble
from sklearn import svm
from sklearn import neural_network
from sklearn import model_selection
from sklearn import metrics
from sklearn.utils.fixes import loguniform
from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_selection import SelectPercentile
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
warnings.filterwarnings("ignore")


def split(data):
    df = data.copy()
    label = df.poi.astype(int)
    feature = df.drop("poi", axis=1)
    return feature, label


def read_data():
    # Read training and testing dataset
    train_data = pd.read_csv("train_data.csv")
    test_data = pd.read_csv("test_features.csv")
    train_data = process_data(train_data, strategy="mean", add_features=True)
    test_data = process_data(test_data, strategy="mean", add_features=True)
    # Scale the data
    train_cols = train_data.columns.tolist()
    test_cols = test_data.columns.tolist()
    feature_scaler = preprocessing.MinMaxScaler()
    train_data = feature_scaler.fit_transform(train_data)
    train_data = pd.DataFrame(train_data, columns=train_cols)
    test_data = feature_scaler.fit_transform(test_data)
    test_data = pd.DataFrame(test_data, columns=test_cols)
    return train_data, test_data


def process_data(data, strategy="mean", add_features=True):
    data = data.query('name!="THE TRAVEL AGENCY IN THE PARK"')
    data = data.query('name!="LOCKHART EUGENE E"')
    drop_feature = ["email_address", "name"]
    data = data.drop(labels=drop_feature, axis=1)
    col = data.columns.tolist()
    imputer = SimpleImputer(missing_values=np.nan, copy=False, strategy=strategy)
    imputer = imputer.fit(data)
    data = imputer.transform(data)
    data = pd.DataFrame(data, columns=col)
    if add_features:
        data = expand_features(data)
    print(f"Shape of data: {data.shape}")
    return data


def expand_features(data):
    data.loc[:, "salary_p"] = data.loc[:, "salary"]/data.loc[:, "total_payments"]
    data.loc[:, "deferral_payments_p"] = data.loc[:, "deferral_payments"]/data.loc[:, "total_payments"]
    data.loc[:, "loan_advances_p"] = data.loc[:, "loan_advances"]/data.loc[:, "total_payments"]
    data.loc[:, "bonus_p"] = data.loc[:, "bonus"]/data.loc[:, "total_payments"]
    data.loc[:, "deferred_income_p"] = data.loc[:, "deferred_income"]/data.loc[:, "total_payments"]
    data.loc[:, "expenses_p"] = data.loc[:, "expenses"]/data.loc[:, "total_payments"]
    data.loc[:, "other_p"] = data.loc[:, "other"]/data.loc[:, "total_payments"]
    data.loc[:, "long_term_incentive_p"] = data.loc[:, "long_term_incentive"]/data.loc[:, "total_payments"]
    data.loc[:, "director_fees_p"] = data.loc[:, "director_fees"]/data.loc[:, "total_payments"]
    data.loc[:, "restricted_stock_deferred_p"] = data.loc[:, "restricted_stock_deferred"]/data.loc[:, "total_stock_value"]
    data.loc[:, "exercised_stock_options_p"] = data.loc[:, "exercised_stock_options"]/data.loc[:, "total_stock_value"]
    data.loc[:, "restricted_stock_p"] = data.loc[:, "restricted_stock"]/data.loc[:, "total_stock_value"]
    data.loc[:, "from_poi_to_this_person_p"] = data.loc[:, "from_poi_to_this_person"]/data.loc[:, "to_messages"]
    data.loc[:, "shared_receipt_with_poi_p"] = data.loc[:, "shared_receipt_with_poi"]/data.loc[:, "to_messages"]
    data.loc[:, "from_this_person_to_poi_p"] = data.loc[:, "from_this_person_to_poi"]/data.loc[:, "from_messages"]
    data.loc[:, "long_term_incentive_p"] = data.loc[:, "long_term_incentive"]/data.loc[:, "total_payments"]
    data.loc[:, "restricted_stock_deferred_p"] = data.loc[:, "restricted_stock_deferred"]/data.loc[:, "total_stock_value"]
    data.loc[:, "from_this_person_to_poi_p"] = data.loc[:, "from_this_person_to_poi"]/data.loc[:, "from_messages"]
    data.drop("long_term_incentive", axis=1)
    data.drop("restricted_stock_deferred", axis=1)
    data.drop("from_this_person_to_poi", axis=1)
    return data


def run_training(train_data, clf, name, folds=5):
    # Add fold column
    train_data.loc[:, "kfold"] = -1
    train_data = train_data.sample(frac=1).reset_index(drop=True)
    label = train_data.poi.astype(int)
    skf = model_selection.StratifiedKFold(n_splits=folds, random_state=914)
    for f, (t_, v_) in enumerate(skf.split(X=train_data, y=train_data.poi)):
        train_data.loc[v_, "kfold"] = f
    # Run cross validation
    dfs = []
    auc_list = []
    for fold in range(folds):
        # Create fold for training and validation dataset
        df_train = train_data[train_data.kfold != fold].reset_index(drop=True)
        df_valid = train_data[train_data.kfold == fold].reset_index(drop=True)
        df_train = df_train.drop("kfold", axis=1)
        df_valid = df_valid.drop("kfold", axis=1)
        x_train, y_train = split(df_train)
        x_valid, y_valid = split(df_valid)
        # Fit the classification model
        clf.fit(x_train, y_train)
        pred = clf.predict_proba(x_valid)[:, 1]
        # Calculate roc auc score
        auc = metrics.roc_auc_score(y_valid, pred)
        auc = round(auc, 4)
        auc_list.append(auc)
        print(f"Fold {fold+1}: {auc}")
        df_valid.loc[:, "{}_pred".format(name)] = pred
        dfs.append(df_valid)
    final_df = pd.concat(dfs)
    print(f"Average AUC: {round(np.mean(auc_list), 4)}")
    final_df.to_csv(f"./prediction/{name}.csv", index=False)
    return clf.best_estimator_


def blending(data, folds):
    names = [
        "LinearSVM",
        "LogisticRegression",
        "RandomForest",
        "XGBoost", 
        "CatBoost", 
        "LGMBoost"
        ]
    classifiers = [
        svm.SVC(probability=True),
        linear_model.LogisticRegression(max_iter=10000),
        ensemble.RandomForestClassifier(),
        XGBClassifier(), 
        CatBoostClassifier(verbose=0), 
        LGBMClassifier(n_estimators=400, silent=True)
    ]
    parameters = [
        # LinearSVM
        {'C': loguniform(1e0, 1e3),
         'gamma': loguniform(1e-4, 1e-3),
         'kernel': ['rbf']
         }, 
        # LogisticRegression
        {}, 
         # RandomForest
        {'bootstrap': [True, False],
         'max_depth': [int(x) for x in range(10, 50)],
         'max_features': ['auto', 'sqrt'],
         'min_samples_leaf': [int(x) for x in range(1, 5)],
         'min_samples_split': [int(x) for x in range(2, 10)],
         'n_estimators': [int(x) for x in range(100, 500, 50)]
         }, 
         # XGBoost
        {'min_child_weight': [1, 5, 10],
         'gamma': loguniform(1e-4, 1e-3),
         'subsample': list(np.linspace(0.5, 1, 100)),
         'colsample_bytree': list(np.linspace(0.6, 1, 10)),
         'max_depth': [int(x) for x in range(3, 11)], 
         'n_estimators': [int(x) for x in range(100, 500, 50)]
        }, 
        # CatBoost
        {'max_depth': [int(x) for x in range(4, 11)], 
         'iterations': [int(x) for x in range(10, 100)]
        }, 
        # LGMBoost
        {}
    ]
    for name, classifier, param_dist in zip(names, classifiers, parameters):
        print(name)
        print("-"*50)
        train_data = data.copy()
        n_iter_search = 30
        rs = RandomizedSearchCV(
            classifier, param_distributions=param_dist, n_iter=n_iter_search, n_jobs=-1)
        best_clf = run_training(train_data, rs, name, folds)
        print("\n")
        joblib.dump(best_clf, f"./model/{name}.bin", compress=5)


def predictor(test_data):
    names = [
        "LinearSVM",
        "LogisticRegression",
        "RandomForest",
        "XGBoost", 
        "CatBoost", 
        "LGMBoost"
        ]
    classifiers = [
        joblib.load("./model/LinearSVM.bin"),
        joblib.load("./model/LogisticRegression.bin"),
        joblib.load("./model/RandomForest.bin"),
        joblib.load("./model/XGBoost.bin"), 
        joblib.load("./model/CatBoost.bin"), 
        joblib.load("./model/LGMBoost.bin")
    ]
    dfs = []
    for name, classifier in zip(names, classifiers):
        pred = classifier.predict_proba(test_data)[:, 1]
        dfs.append(pred)
    dfs = np.array(dfs)
    # prob = (dfs[0]+dfs[1]+dfs[2]+dfs[3]+2*dfs[4]+dfs[5])/7
    prob = np.mean(dfs, axis=0)
    # submit = pd.read_csv("sample_submission.csv")
    # submit['poi'] = prob
    # submit.to_csv('Submission_20201107.csv', index=False)


if __name__ == "__main__":
    TRAIN = True
    train_data, test_data = read_data()
    if TRAIN:
        # Start blending
        blending(train_data, folds=10)
    # Read prediction csv file into dataframe
    files = glob.glob("./prediction/*.csv")
    df = None
    for f in files:
        if df is None:
            df = pd.read_csv(f)
        else:
            temp_df = pd.read_csv(f)
            df = pd.concat([df, temp_df.iloc[:, -1]], axis=1)
    targets = df.poi.values
    pred_cols = [
        "LinearSVM_pred", "LogisticRegression_pred", "RandomForest_pred", "XGBoost_pred", 
        "CatBoost_pred", "LGMBoost_pred"]
    for col in pred_cols:
        auc = metrics.roc_auc_score(targets, df[col].values)
        print(f"{col}, overall auc={auc}")
    print(metrics.roc_auc_score(targets, np.mean(df[pred_cols].values, axis=1)))
    svm_pred = df.LinearSVM_pred.values
    lrc_pred = df.LogisticRegression_pred.values
    rfc_pred = df.RandomForest_pred.values
    xgb_pred = df.XGBoost_pred.values
    cat_pred = df.CatBoost_pred.values
    lgb_pred = df.LGMBoost_pred.values
    avg_pred = (svm_pred+lrc_pred+rfc_pred+xgb_pred+2*cat_pred+lgb_pred)/7
    print(metrics.roc_auc_score(targets, avg_pred))
    predictor(test_data)