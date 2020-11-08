import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
warnings.filterwarnings("ignore")


def read_data():
    train_data = pd.read_csv("./data/train_data.csv")
    test_data = pd.read_csv("./data/test_features.csv")
    train_data = preprocess_data(train_data, mode="train")
    test_data = preprocess_data(test_data, mode="test")
    train_feature, train_label = target_feature_split(train_data)
    train_cols = train_feature.columns.tolist()
    test_cols = test_data.columns.tolist()
    feature_scaler = preprocessing.MinMaxScaler()
    train_feature = feature_scaler.fit_transform(train_feature)
    train_feature = pd.DataFrame(train_feature, columns=train_cols)
    test_feature = feature_scaler.fit_transform(test_data)
    test_feature = pd.DataFrame(test_feature, columns=test_cols)
    return train_feature, test_feature, train_label

def preprocess_data(data, mode="train"):
    if mode == "train":
        # Drop outliers
        data = data.query('name!="THE TRAVEL AGENCY IN THE PARK"')
        data = data.query('name!="LOCKHART EUGENE E"')
        data = data.query('name!="TOTAL"')
        data.loc[:, "poi"] = data["poi"].map(int)
    # All columns name
    columns = [
        'name', 'bonus', 'deferral_payments', 'deferred_income', 'director_fees', 
        'email_address', 'exercised_stock_options', 'expenses', 'from_messages', 
        'from_poi_to_this_person', 'from_this_person_to_poi', 'loan_advances', 
        'long_term_incentive', 'other', 'poi', 'restricted_stock', 'restricted_stock_deferred', 
        'salary', 'shared_receipt_with_poi', 'to_messages', 'total_payments', 'total_stock_value']
    # print("# of all features (include label): ", len(columns))
    financial_features = [
        'salary','bonus', 'exercised_stock_options', 'restricted_stock', 'shared_receipt_with_poi',
        'total_payments', 'expenses', 'total_stock_value', 'deferred_income', 'long_term_incentive', 
        'deferral_payments', 'director_fees', 'loan_advances', 'restricted_stock_deferred']
    # print("# of financial features: ", len(financial_features))
    behavioral_features = [
        'to_messages', 'from_messages', 'from_poi_to_this_person', 'from_this_person_to_poi', 'other']
    # print("# of behavioral features: ", len(behavioral_features))
    # Drop name and email address feature
    data.drop("name", inplace=True, axis=1)
    data.drop("email_address", inplace=True, axis=1)
    data = data.apply(lambda x: x.fillna(x.mean()),axis=0)
    data = add_features(data)
    return data


def add_features(data):
    data.loc[:, "fraction_from_poi"] = data.loc[:, "from_poi_to_this_person"].divide(data.loc[:, "to_messages"], fill_value=0)
    data.loc[:, "fraction_to_poi"] = data.loc[:, "from_this_person_to_poi"].divide(data.loc[:, "from_messages"], fill_value=0)
    data.loc[:, "fraction_from_poi"] = data.loc[:, "fraction_from_poi"].fillna(0.0)
    data.loc[:, "fraction_to_poi"] = data.loc[:, "fraction_to_poi"].fillna(0.0)
    # http://www.yannispappas.com/Fraud-Detection-Using-Machine-Learning/
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
    # https://medium.com/@williamkoehrsen/machine-learning-with-python-on-the-enron-dataset-8d71015be26d
    data.loc[:, 'to_poi_ratio'] = data.loc[:, 'from_poi_to_this_person']/data.loc[:, 'to_messages']
    data.loc[:, 'from_poi_ratio'] = data.loc[:, 'from_this_person_to_poi']/data.loc[:, 'from_messages']
    data.loc[:, 'shared_poi_ratio'] = data.loc[:, 'shared_receipt_with_poi']/data.loc[:, 'to_messages']
    data.loc[:, 'bonus_to_salary'] = data.loc[:, 'bonus'] / data.loc[:, 'salary']
    data.loc[:, 'bonus_to_total'] = data.loc[:, 'bonus'] / data.loc[:, 'total_payments']
    # Drop columns
    data.drop("long_term_incentive", axis=1, inplace=True)
    data.drop("restricted_stock_deferred", axis=1, inplace=True)
    data.drop("from_this_person_to_poi", axis=1, inplace=True)
    return data


def type_of_col(data):
    df = data.copy()
    int_features = []
    float_features = []
    object_features = []
    for dtype, feature in zip(df.dtypes, df.columns):
        if dtype == 'float64':
            float_features.append(feature)
        elif dtype == 'int64':
            int_features.append(feature)
        else:
            object_features.append(feature)
    print(f'{len(int_features)} Integer Features : {int_features}\n')
    print(f'{len(float_features)} Float Features : {float_features}\n')
    print(f'{len(object_features)} Object Features : {object_features}')
    return int_features, float_features, object_features


def target_feature_split(data):
    df = data.copy()
    label = df.poi.astype(int)
    feature = df.drop("poi", axis=1)
    return feature, label


def show_distribution(data, nrows=4, ncols=5):
    fig, axes = plt.subplots(figsize=(15, 10))
    for i, column in enumerate(data.columns):
        plt.subplot(nrows, ncols, i+1)
        sns.distplot(data.loc[:, column], ax=plt.gca())
        plt.xlabel(column)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    train_feature, test_feature, train_label = read_data()