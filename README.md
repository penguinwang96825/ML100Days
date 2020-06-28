# ML100Days
ML100Days is a project to keep studying and coding for 100 days.

# Exploratory Data Analysis (EDA)
## Data Type ([Day 007](https://github.com/penguinwang96825/ML100Days/blob/master/homework/Day007/Day_007_HW.ipynb))
In general, there are four data types in dataframe, that is `int`, `float`, `object`, `datetime`.
```python
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
```

## Visualization ([Day 008](https://github.com/penguinwang96825/ML100Days/blob/master/homework/Day008/Day_008_HW.ipynb))
1. *Normal distribution* describes continuous data which have a symmetric distribution, with a characteristic 'bell' shape.
2. *Binomial distribution* describes the distribution of binary data from a finite sample. Thus it gives the probability of getting r events out of n trials.
3. *Poisson distribution* describes the distribution of binary data from an infinite sample. Thus it gives the probability of getting r events in a population.
```python
def plot_category_chart(df, feature, rotation=90, ascending=False, title=None, x_name=None, figsize=(10, 4)):
    """
    Parameters:
    ----------
        df: DataFrame
        feature: str
        rotation: int
        ascending: bool
        title: str
        x_name: list
        figsize: tuple
    
    Returns:
    -------
        matplotlib chart
    """
    plt.figure(figsize=figsize)
    if ascending == True:
        chart = sns.countplot(df[feature], order=df[feature].value_counts().index)
    else: 
        chart = sns.countplot(df[feature])
    chart.set_xticklabels(chart.get_xticklabels(), rotation=rotation, ha="right")
    plt.grid(axis="y")
    plt.title(title)
    if x_name is not None:
        plt.xticks(np.arange(len(x_name)), x_name)
    plt.tight_layout()
    plt.show()
```

## Outlier Detection ([Day 009](https://github.com/penguinwang96825/ML100Days/blob/master/homework/Day009/Day_009_HW.ipynb))
### Boxplot
Discover outliers with visualization tools, such as boxplot and scatterplot.
```python
# Select numerical columns
numeric_columns = list(app_train.columns[
    list(app_train.dtypes.isin([np.dtype('int'), np.dtype('float')]))])

# Remove the columns which only have two unique values
numeric_columns = list(app_train[numeric_columns].columns[
    list(app_train[numeric_columns].apply(lambda x:len(x.unique())!=2))])
print("Numbers of remain columns: %i" % len(numeric_columns))

# Plot columns' boxplot
plt.figure(figsize=(16, 56))
for i, col in enumerate(numeric_columns):
    plt.subplot(13, 5, i+1)
    app_train.boxplot(col)
plt.tight_layout()
plt.show()
```

### Emprical Cumulative Density Plot (ECDF)
**Why is the Empirical Cumulative Distribution Useful in Exploratory Data Analysis?**
The empirical CDF is useful because

1. It approximates the true CDF well if the sample size (the number of data) is large, and knowing the distribution is helpful for statistical inference.
2. A plot of the empirical CDF can be visually compared to known CDFs of frequently used distributions to check if the data came from one of those common distributions.
3. It can visually display “how fast” the CDF increases to 1; plotting key quantiles like the quartiles can be useful to “get a feel” for the data.

```python
def plot_ecdf(df, column):
    """
    Parameters
        df: DataFrame
        column: str
    """
    cdf = df[column].value_counts().sort_index().cumsum()

    plt.figure(figsize=(15, 4))
    plt.plot(list(cdf.index), cdf/cdf.max())
    plt.grid()
    plt.xlabel('Value')
    plt.ylabel('ECDF')
    plt.ylim([-0.05, 1.05])
    plt.show()
```

### Deal with Outliers
During data analysis when you detect the outlier one of most difficult decision could be how one should deal with the outlier. Should they remove them or correct them?
1. Clipping
```python
df[column] = df[column].clip(500, 2500)
```
2. Deleting
```python
keep_indexs = (df[column]> 500) & (df[column]< 2500)
df = df[keep_indexs]
```
3. IQR
```python
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
df_out = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
```
The above code will remove the outliers from the dataset.

### Reduce Skewness
1. log1p: 
```python
df[column] = np.log1p(df[column])
```
2. boxcox: (boxcox can not have negative value)
```python
from scipy import stats
df_fixed[column] = df_fixed[column].apply(lambda x: x+1 if x <= 0 else x)
df_fixed[column] = stats.boxcox(df_fixed[column])[0]
```

## Correlation Label Plot
```python
cc = df.corr()['TARGET'].sort_values()
cc_min = cc.index.to_list()[0:15]
cc_max = cc.index.to_list()[-16:-1]

plt.figure(figsize=(15, 15))
try: 
    for i in range(15):
        plt.subplot(5, 3, i+1)
        sns.distplot(df.loc[df["TARGET"] == 0][cc_min[i]], kde=False, label="Negative")
        sns.distplot(df.loc[df["TARGET"] == 1][cc_min[i]], kde=False, label="Positive")
        plt.xticks(rotation=90)
        plt.title("{}".format(cc_min[i]))
        plt.legend()
except: 
    pass
plt.tight_layout()
plt.show()

plt.figure(figsize=(15, 15))
try: 
    for i in range(15):
        plt.subplot(5, 3, i+1)
        sns.distplot(df.loc[df["TARGET"] == 0][cc_max[i]], kde=False, label="Negative")
        sns.distplot(df.loc[df["TARGET"] == 1][cc_max[i]], kde=False, label="Positive")
        plt.xticks(rotation=90)
        plt.title("{}".format(cc_max[i]))
except: 
    pass
plt.tight_layout()
plt.show()
```

# Feature Engineering
## Label Encoding (Ordered Discretized Data) ([Day 024](https://github.com/penguinwang96825/ML100Days/blob/master/homework/Day024/Day_024_HW.ipynb))
This approach is very simple and it involves converting each value in a column to a number.
```python
from sklearn.preprocessing import LabelEncoder
df[column] = LabelEncoder().fit_transform(df[column])
```

## One-Hot Encoding (Unordered Discretized Data) ([Day 024](https://github.com/penguinwang96825/ML100Days/blob/master/homework/Day024/Day_024_HW.ipynb))
In this strategy, each category value is converted into a new column and assigned a 1 or 0 (notation for true/false) value to the column.
```python
df = pd.get_dummies(df)
```

## Mean Encoding Version 1 ([Day 025](https://github.com/penguinwang96825/ML100Days/blob/master/homework/Day025/Day_025_HW.ipynb))
Mean Encoding is a simple preprocessing scheme for high-cardinality categorical data that allows this class of attributes to be used in predictive models such as neural networks, linear and logistic regression. The proposed method is based on a well-established statistical method (empirical Bayes) that is straightforward to implement as an in-database procedure. Furthermore, for categorical attributes with an inherent hierarchical structure, like ZIP codes, the preprocessing scheme can directly leverage the hierarchy by blending statistics at the various levels of aggregation.
```python
# Reference from http://www.jiangdongzml.com/2018/01/31/Catergorical_Attributes/
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from itertools import product

class MeanEncoder:
    def __init__(
        self, 
        categorical_features, 
        n_splits=20, 
        target_type='classification', 
        prior_weight_func=None):
        """
        :param categorical_features: list of str, the name of the categorical columns to encode

        :param n_splits: the number of splits used in mean encoding

        :param target_type: str, 'regression' or 'classification'

        :param prior_weight_func:
        a function that takes in the number of observations, and outputs prior weight
        when a dict is passed, the default exponential decay function will be used:
        k: the number of observations needed for the posterior to be weighted equally as the prior
        f: larger f --> smaller slope
        """

        self.categorical_features = categorical_features
        self.n_splits = n_splits
        self.learned_stats = {}

        if target_type == 'classification':
            self.target_type = target_type
            self.target_values = []
        else:
            self.target_type = 'regression'
            self.target_values = None

        # Calculate smoothing factor: 
        # smoothing_factor = 1 / (1 + np.exp(- (counts - min_samples_leaf) / smoothing_slope))
        if isinstance(prior_weight_func, dict):
            self.prior_weight_func = eval(
                'lambda x: 1 / (1 + np.exp(- (x - k) / f))', dict(prior_weight_func, np=np))
        elif callable(prior_weight_func):
            self.prior_weight_func = prior_weight_func
        else:
            self.prior_weight_func = lambda x: 1 / (1 + np.exp(- (x - 2) / 1))

    @staticmethod
    def mean_encode_subroutine(X_train, y_train, X_test, variable, target, prior_weight_func):
        X_train = X_train[[variable]].copy()
        X_test = X_test[[variable]].copy()

        if target is not None:
            nf_name = '{}_pred_{}'.format(variable, target)
            X_train['pred_temp'] = (y_train == target).astype(int)
        else:
            nf_name = '{}_pred'.format(variable)
            X_train['pred_temp'] = y_train
        prior = X_train['pred_temp'].mean()

        col_avg_y = X_train.groupby(by=variable, axis=0)['pred_temp'].agg(mean="mean", beta="size")
        col_avg_y['beta'] = prior_weight_func(col_avg_y['beta'])
        # Calculate smoothing_mean: 
        # smoothing_mean = smoothing_factor * estimated_mean + (1 - smoothing_factor) * overall_mean
        col_avg_y[nf_name] = col_avg_y['beta'] * col_avg_y['mean'] + (1 - col_avg_y['beta']) * prior
        col_avg_y.drop(['beta', 'mean'], axis=1, inplace=True)

        nf_train = X_train.join(col_avg_y, on=variable)[nf_name].values
        nf_test = X_test.join(col_avg_y, on=variable).fillna(prior, inplace=False)[nf_name].values

        return nf_train, nf_test, prior, col_avg_y

    def fit_transform(self, X, y):
        """
        :param X: pandas DataFrame, n_samples * n_features
        :param y: pandas Series or numpy array, n_samples
        :return X_new: the transformed pandas DataFrame containing mean-encoded categorical features
        """
        X_new = X.copy()
        if self.target_type == 'classification':
            skf = StratifiedKFold(self.n_splits)
        else:
            skf = KFold(self.n_splits)

        if self.target_type == 'classification':
            self.target_values = sorted(set(y))
            self.learned_stats = {'{}_pred_{}'.format(variable, target): [] for variable, target in
                                  product(self.categorical_features, self.target_values)}
            for variable, target in product(self.categorical_features, self.target_values):
                nf_name = '{}_pred_{}'.format(variable, target)
                X_new.loc[:, nf_name] = np.nan
                for large_ind, small_ind in skf.split(y, y):
                    nf_large, nf_small, prior, col_avg_y = MeanEncoder.mean_encode_subroutine(
                        X_new.iloc[large_ind], 
                        y.iloc[large_ind], 
                        X_new.iloc[small_ind], 
                        variable, 
                        target, 
                        self.prior_weight_func)
                    X_new.iloc[small_ind, -1] = nf_small
                    self.learned_stats[nf_name].append((prior, col_avg_y))
        else:
            self.learned_stats = {'{}_pred'.format(variable): [] for variable in self.categorical_features}
            for variable in self.categorical_features:
                nf_name = '{}_pred'.format(variable)
                X_new.loc[:, nf_name] = np.nan
                for large_ind, small_ind in skf.split(y, y):
                    nf_large, nf_small, prior, col_avg_y = MeanEncoder.mean_encode_subroutine(
                        X_new.iloc[large_ind], 
                        y.iloc[large_ind], 
                        X_new.iloc[small_ind], 
                        variable, 
                        None, 
                        self.prior_weight_func)
                    X_new.iloc[small_ind, -1] = nf_small
                    self.learned_stats[nf_name].append((prior, col_avg_y))
        return X_new

    def transform(self, X):
        """
        :param X: pandas DataFrame, n_samples * n_features
        :return X_new: the transformed pandas DataFrame containing mean-encoded categorical features
        """
        X_new = X.copy()

        if self.target_type == 'classification':
            for variable, target in product(self.categorical_features, self.target_values):
                nf_name = '{}_pred_{}'.format(variable, target)
                X_new[nf_name] = 0
                for prior, col_avg_y in self.learned_stats[nf_name]:
                    X_new[nf_name] += X_new[[variable]].join(
                        col_avg_y, on=variable).fillna(prior, inplace=False)[
                        nf_name]
                X_new[nf_name] /= self.n_splits
        else:
            for variable in self.categorical_features:
                nf_name = '{}_pred'.format(variable)
                X_new[nf_name] = 0
                for prior, col_avg_y in self.learned_stats[nf_name]:
                    X_new[nf_name] += X_new[[variable]].join(
                        col_avg_y, on=variable).fillna(prior, inplace=False)[
                        nf_name]
                X_new[nf_name] /= self.n_splits

        return X_new
```

Usage: 
```python
cat = ['Sex', 'Cabin', 'Embarked']
X_train = df_train[cat].copy()
y_train = df_train['Survived'].copy()
me = MeanEncoder(categorical_features=cat, n_splits=5, target_type='classification')
X_train = me.fit_transform(X_train, y_train)
X_train = X_train.drop(cat, axis=1)
```

## Mean Encoding Version 2 ([Day 025](https://github.com/penguinwang96825/ML100Days/blob/master/homework/Day025/Day_025_HW.ipynb))
Using the mean of the target values to replace the original categorical features. One thing need to be careful with this method is really easy to overfit the data even after smoothing. If we only have a little dataset and we accidently chose an extreme value will end up getting a mean value with deviation. So we add in the counts of the values as reliability when using mean encoding.
```python
# df only contains object-type feature
data = pd.concat([df[:train_num], train_Y], axis=1)
for c in df.columns:
    mean_df = data.groupby([c])['Survived'].mean().reset_index()
    mean_df.columns = [c, f'{c}_mean']
    data = pd.merge(data, mean_df, on=c, how='left')
    data = data.drop([c], axis=1)
data = data.drop(['Survived', 'Name_mean', 'Ticket_mean'], axis=1)
```

# Feature Importance
1. Feature Importance
```python
estimator = RandomForestClassifier()
estimator.fit(df.values, train_Y)
feats = pd.Series(data=estimator.feature_importances_, index=df.columns)
feats = feats.sort_values(ascending=False)
```

2. Permutation Importance
- Permutation importance is calculated after a model has been fitted.
- If I randomly shuffle a single column of the validation data, leaving the target and all other columns in place, how would that affect the accuracy of predictions in that now-shuffled data?
- Randomly re-ordering a single column should cause less accurate predictions, since the resulting data no longer corresponds to anything observed in the real world. Model accuracy especially suffers if we shuffle a column that the model relied on heavily for predictions.

**Steps**
1. Get a trained model.
2. Shuffle the values in a single column, make predictions using the resulting dataset. Use these predictions and the true target values to calculate how much the loss function suffered from shuffling. That performance deterioration measures the importance of the variable you just shuffled.
3. Return the data to the original order (undoing the shuffle from step 2). Now repeat step 2 with the next column in the dataset, until you have calculated the importance of each column.
```python
from sklearn.model_selection import train_test_split

data = pd.read_csv(data_path + 'titanic_train.csv')
y = data['Survived']
data = data.drop(labels=["Survived"], axis=1)

LEncoder = LabelEncoder()
MMEncoder = MinMaxScaler()
for c in data.columns:
    data[c] = data[c].fillna(-1)
    if data[c].dtype == 'object':
        data[c] = LEncoder.fit_transform(list(data[c].values))
    data[c] = MMEncoder.fit_transform(data[c].values.reshape(-1, 1))

feature_names = [i for i in data.columns if data[i].dtype in [np.dtype("int"), np.dtype("float")]]
X = data[feature_names]
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=17)
my_model = RandomForestClassifier(n_estimators=100,
                                  random_state=17).fit(train_X, train_y)

import eli5
from eli5.sklearn import PermutationImportance

perm = PermutationImportance(my_model, random_state=1).fit(val_X, val_y)
eli5.show_weights(perm, feature_names=val_X.columns.tolist())
```
Reference from [here](https://www.kaggle.com/dansbecker/permutation-importance?utm_medium=email&utm_source=mailchimp&utm_campaign=ml4insights).

# Metrics 
## Precision ([Day 032](https://github.com/penguinwang96825/ML100Days/blob/master/homework/Day032/Day_032_HW.ipynb))
Precision attempts to answer the following question: 
What proportion of positive identifications was actually correct?

## Recall ([Day 032](https://github.com/penguinwang96825/ML100Days/blob/master/homework/Day032/Day_032_HW.ipynb))
Recall attempts to answer the following question: 
What proportion of actual positives was identified correctly?

## F Score ([Day 036](https://github.com/penguinwang96825/ML100Days/blob/master/homework/Day036/Day_036_HW.ipynb))
```python
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

def f2_score(y_true, y_pred, beta):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    nominator = (1+beta**2)*(precision*recall)
    denominator = beta**2*precision+recall
    return nominator/denominator
```

## Bias-Variance Tradeoff ([Day 033](https://github.com/penguinwang96825/ML100Days/blob/master/homework/Day033/Day_033_HW.ipynb))
In statistics and machine learning, the bias–variance tradeoff is the property of a set of predictive models whereby models with a lower bias in parameter estimation have a higher variance of the parameter estimates across samples, and vice versa.

- Low Bias:
![](http://i1.bangqu.com/j/news/20180123/5bc93e9fbee64d909795afa4f52b16dd.jpeg)

- High Bias:
![](http://i1.bangqu.com/j/news/20180123/357b5ba39ad042aa9d1a96eed5c2e75d.png)

## Learning Curve ([Day 033](https://github.com/penguinwang96825/ML100Days/blob/master/homework/Day033/Day_033_HW.ipynb))
A learning curve is a relationship of the duration or the degree of effort invested in learning and experience with the resulting progress, considered as an exploratory discovery process.

![](http://i1.bangqu.com/j/news/20180123/89edbff69a834fc6a09357730be3d37b.png)
![](http://i1.bangqu.com/j/news/20180123/bfe60571bf784dd7a43c4f44c49e3b28.png)

Reference from [here](http://bangqu.com/yjB839.html).

## Object Function ([Day 033](https://github.com/penguinwang96825/ML100Days/blob/master/homework/Day033/Day_033_HW.ipynb))
1. Regression Loss Functions

 - Mean Squared Error Loss
 - Mean Squared Logarithmic Error Loss
 - Mean Absolute Error Loss

2. Binary Classification Loss Functions
 - Binary Cross-Entropy
 - Hinge Loss
 - Squared Hinge Loss

3. Multi-Class Classification Loss Functions
 - Multi-Class Cross-Entropy Loss
 - Sparse Multiclass Cross-Entropy Loss
 - Kullback Leibler Divergence Loss

# Train Test Split
Solve unbalanced data problem. ([Day 034](https://github.com/penguinwang96825/ML100Days/blob/master/homework/Day034/Day_034_HW.ipynb))
```python
from sklearn.model_selection import train_test_split
y_1_index, y_0_index = np.where(y==1)[0], np.where(y==0)[0]
train_data_y1, test_data_y1, train_label_y1, test_label_y1 = train_test_split(
    X[y_1_index], y[y_1_index], test_size=10)
train_data_y0, test_data_y0, train_label_y0, test_label_y0 = train_test_split(
    X[y_0_index], y[y_0_index], test_size=10)
x_train, y_train = np.concatenate(
    [train_data_y1, train_data_y0]), np.concatenate([train_label_y1, train_label_y0])
x_test, y_test = np.concatenate(
    [test_data_y1, test_data_y0]), np.concatenate([test_label_y1, test_label_y0])
```

# Modelling
## Regression Model ([Day 037](https://github.com/penguinwang96825/ML100Days/blob/master/homework/Day037/Day_037_HW.ipynb))
1. Can linear model solve non-linear data?
In linear regression, the relationships are modeled using linear predictor functions whose unknown model parameters are estimated from the data. The difference between linear and nonlinear regression models isn’t as straightforward as it sounds. You’d think that linear equations produce straight lines and nonlinear equations model curvature. Unfortunately, that’s not correct. Both types of models can fit curves to your data—so that’s not the defining characteristic. A linear regression model follows a very particular form. In statistics, a regression model is linear when all terms in the model are one of the following:
- The constant
- A parameter multiplied by an independent variable

2. Is there any hypothesis for linear model?
It is customary in econometrics to state the assumptions of the regression model in terms of the random error *e*. For future reference the assumptions are named SR1-SR6, "SR" denoting "simple regression." 
 - The value of *y*, for each value of *x*, is <img src="https://render.githubusercontent.com/render/math?math=y = \beta_1+\beta_2x+e">
 - The average value of the random error *e* is $E(e) = 0$ since we assume that $y = \beta_1+\beta_2x$
 - The variance of the random error *e* is $var(e) = \sigma^2 = var(y)$ since *y* and *e* differ only by a constant, which doesn’t change the variance. 
 - The covariance between any pair of random errors, $e_i$ and $e_j$ is $cov(e_i, e_j) = cov(y_i, y_j) = 0$.  If the values of *y* are statistically independent, then so are the random errors *e*, and vice versa.
 - The variable *x* is not random and must take at least two different values. 
 - (optional) The values of *e* are normally distributed about their mean $e ~ N(0, \sigma^2)$ if the values of *y* are normally distributed, and vice versa.
