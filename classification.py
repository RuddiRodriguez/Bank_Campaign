# %%
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import make_column_selector as selector
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
import warnings
from sklearn import metrics
import numpy as np
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# %%
df = pd.read_csv('../Data/Bank-full.csv', sep=';')
# %%
df.columns
# %%
df.head()
# %%
df.info()
# %%
x_cols = [c for c in df if c not in [['month']]]
df = df[x_cols]

df.y = LabelEncoder().fit_transform(df.y)
# %%
train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
# %%
X_train, y_train = df.drop('y', axis=1), df['y']
""" #%%
# Remove the target column and the phone number
x_cols = [c for c in df if c not in [target and other columns to drop]]
# Column types are defaulted to floats
X = (
    df
    .drop(["target"], axis=1)
    
binary_features = ["binary features list]
categorical_features = [categorical features  list]
X[binary_features] = X[binary_features].astype("bool")

# Categorical features can't be set all at once
for f in categorical_features:
    X[f] = X[f].astype("category") """

# %%
numeric_features = ['age', 'balance']
categorical_features = ['marital']
numeric_transformer = Pipeline(steps=[
    ('impute', SimpleImputer()),
    ('scale', StandardScaler())])

categorical_transformer = OneHotEncoder(handle_unknown='ignore')

t = [('num', numeric_transformer, selector(dtype_include='int64')),
     ('cat', categorical_transformer, selector(dtype_include=['object', 'bool']))]
preprocessor = ColumnTransformer(transformers=t, remainder='drop')
# %%
df_processed = preprocessor.fit_transform(df.iloc[0:30])
df_processed.shape
# %%
df.info()
# %%
preprocessor
# %%
# The full pipeline as a step in another pipeline with an estimator as the final step

full_pipeline_parameters = [('preprocessor_pipeline', preprocessor),
                            ('model', LogisticRegression(class_weight='balanced'))]

full_pipeline_m = Pipeline(steps=full_pipeline_parameters)

log_parameters_ = {'preprocessor_pipeline__num__impute__strategy': ['mean', 'median', 'most_frequent'],
                   # 'preprocessor__pca__n_components':[2],
                   'model': [LogisticRegression()],
                   'model__penalty': ['l1', 'l2'],
                   'model__C': np.logspace(0, 4, 10)
                   }
SVC_parameters_ = {'preprocessor_pipeline__num__impute__strategy': ['mean', 'median', 'most_frequent'],
                   # 'preprocessor__pca__n_components':[2],
                  'model': [SVC(kernel="rbf")],
                  'model__C': [0.5, 1, 5, 10, 30, 40, 50, 75, 100],
                  'model__gamma': [0.05, 0.07, 0.1, 0.5, 1]
                   }
RF_parameters_ = {'preprocessor_pipeline__num__impute__strategy': ['mean', 'median', 'most_frequent'],
                  'preprocessor__pca__n_components': [2, 5],
                  'model': [RandomForestClassifier()],
                  'model__n_estimators': [10, 100, 1000],
                  "model__max_depth": [5, 8, 30, None],
                  "model__min_samples_leaf": [1, 2, 5, 10],
                  "model__max_leaf_nodes": [2, 5, 10]
                  }
param_grid = [log_parameters_, RF_parameters_, SVC_parameters_]

scoring = ['precision_macro', 'recall_macro', 'f1_macro', 'balanced_accuracy']

grid_search = GridSearchCV(full_pipeline_m, param_grid, cv=3,
                           n_jobs=-1, verbose=1, scoring=scoring, refit='recall_macro')
# %%   #
grid_search.fit(X_train, y_train)
print(grid_search.best_params_)
# %%
from sklearn import set_config

set_config(display='diagram')
grid_search
# %%
