!mamba install -qy pandas==1.3.4 numpy==1.21.4 seaborn==0.9.0 matplotlib==3.5.0 scikit-learn==0.20.1
!pip install lazypredict
!pip install scikit-learn==0.23.1

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

import lazypredict
from lazypredict.Supervised import LazyClassifier
from lazypredict.Supervised import LazyRegressor
from sklearn.model_selection import train_test_split

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')

sns.set_context('notebook')
sns.set_style('white')

adm = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IND-GPXX0JD1EN/adm_data.csv")

#check sample values and column names
adm.head()
list(adm.columns)

# Rename columns that have extra trailing whitespace
adm.rename(columns={"Chance of Admit ": "chance-of-adm", "LOR ": "LOR"}, errors="raise", inplace=True)
adm.info()

# Convert variable 'Research' to take type 'category' because it's a binary variable
adm['Research'] = adm.loc[:, 'Research'].astype('category')

# Check for NAs
adm.isna().sum()

X = adm.drop(['Serial No.', 'chance-of-adm'], axis=1)
y = np.array(adm['chance-of-adm'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

reg = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
models, preds = reg.fit(X_train, X_test, y_train, y_test)
print(models)

# Normalize the metrics for scoring:
models['R-Squared Score'] = (models['R-Squared'] - models['R-Squared'].min()) / (models['R-Squared'].max() - models['R-Squared'].min())
models['RMSE Score'] = 1 - (models['RMSE'] - models['RMSE'].min()) / (models['RMSE'].max() - models['RMSE'].min())
models['Time Taken Score'] = 1 - (models['Time Taken'] - models['Time Taken'].min()) / (models['Time Taken'].max() - models['Time Taken'].min())

# Calculate the composite score
models['Composite Score'] = (models['R-Squared Score'] + models['RMSE Score'] + models['Time Taken Score']) / 3

# Add the model names (from the index) to the DataFrame
models['Model'] = models.index

# Sort models by Composite Score
best_model = models.sort_values('Composite Score', ascending=False).iloc[0]

# Output the best model
print(f"The best model is: {best_model['Model']}")
print(f"Adjusted R-Squared: {best_model['Adjusted R-Squared']}")
print(f"R-Squared: {best_model['R-Squared']}")
print(f"RMSE: {best_model['RMSE']}")
print(f"Time Taken: {best_model['Time Taken']}")
