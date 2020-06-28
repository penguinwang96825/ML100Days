# ML100Days
ML100Days is a project to keep studying and coding for 100 days.

# Exploratory Data Analysis (EDA)
## Data Type
In general, there are four data types in dataframe, that is `int`, `float`, `object`, `datetime`.
<details>
    <summary>Show code.</summary>
        <pre>
            <code>
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
            </code>
        </pre>
</details>

