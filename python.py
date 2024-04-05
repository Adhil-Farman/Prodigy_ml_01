@@ -0,0 +1,67 @@
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

df = pd.read_csv('Data/df.csv')

df['SalePrice'].describe()

Y = df['SalePrice']

features = df[['GrLivArea', 'BedroomAbvGr', 'FullBath', 'HalfBath', 'TotRmsAbvGrd']]

plt.title('Best fit distribution of SalePrice')
sns.histplot(Y, kde=True, stat='density', color='blue', bins=30)

plt.show()

# Log transformation
y_log = np.log1p(Y)

# Plot the transformed data
plt.title('Log-Transformed SalePrice')
sns.histplot(y_log, kde=True, stat='density', color='blue', bins=30)
plt.show()

features = ['GrLivArea', 'BedroomAbvGr', 'FullBath', 'HalfBath', 'TotRmsAbvGrd']

test_normality = lambda x: stats.shapiro(x.fillna(0))[1] < 0.01
normal = pd.DataFrame(df[features])
normal = normal.apply(test_normality)
print(normal.all())

fig, axes = plt.subplots(nrows=len(features), ncols=1, figsize=(4, 2 * len(features)))

# Plot histograms for each feature
for i, feature in enumerate(features):
    sns.histplot(df[feature], kde=True, stat='density', color='blue', bins=30, ax=axes[i])
    axes[i].set_title(feature)

plt.tight_layout()
plt.show()

fig, axes = plt.subplots(nrows=len(features), ncols=1, figsize=(4, 2 * len(features)))

# Plot histograms for each feature
for i, feature in enumerate(features):
    sns.histplot(df[feature], kde=True, stat='density', color='blue', bins=30, ax=axes[i])
    axes[i].set_title(feature)

plt.tight_layout()
plt.show()


GrLivArea_log = np.log1p(df['GrLivArea'])

standard = df[df['SalePrice'] < 200000]
pricey = df[df['SalePrice'] >= 200000]

diff = pd.DataFrame()
diff['feature'] = features
diff['difference'] = [(pricey[f].fillna(0.).mean() - standard[f].fillna(0.).mean())/(standard[f].fillna(0.).mean())
                      for f in features]

sns.barplot(data=diff, x='feature', y='difference')
x=plt.xticks(rotation=90)
