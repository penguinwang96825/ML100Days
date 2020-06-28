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
numeric_columns = list(app_train.columns[list(app_train.dtypes.isin([np.dtype('int'), np.dtype('float')]))])

# Remove the columns which only have two unique values
numeric_columns = list(app_train[numeric_columns].columns[list(app_train[numeric_columns].apply(lambda x:len(x.unique())!=2 ))])
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
2. Deleting
3. IQR
```python
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
df_out = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
```
The above code will remove the outliers from the dataset.