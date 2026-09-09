import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
housing = pd.read_csv("housing.csv")
housing['income_cat'] = pd.cut(housing['median_income'], bins=[0., 1.5, 3.0, 4.5, 6., np.inf], labels=[1, 2, 3, 4, 5])
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing['income_cat']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
housing = strat_train_set.copy()
housing_labels = housing["median_house_value"].copy()
housing = housing.drop("median_house_value", axis=1)
num_attribs = housing.drop("ocean_proximity", axis=1).columns.tolist()
cat_attribs = ["ocean_proximity"]

num_pipeline = Pipeline([
("imputer", SimpleImputer(strategy="median")),
("scaler", StandardScaler()),
])
cat_pipeline = Pipeline([
# ("ordinal", OrdinalEncoder()) # Use this if you prefer ordinal encoding
("onehot", OneHotEncoder(handle_unknown="ignore"))
])
full_pipeline = ColumnTransformer([
("num", num_pipeline, num_attribs),
("cat", cat_pipeline, cat_attribs),
])
housing_prepared = full_pipeline.fit_transform(housing)

#linear regression
lin_red=LinearRegression()
lin_red.fit(housing_prepared, housing_labels)
#tree regression
tree_reg=DecisionTreeRegressor(random_state=42)
tree_reg.fit(housing_prepared, housing_labels)
#random forest regression
forest_reg=RandomForestRegressor(random_state=42)
forest_reg.fit(housing_prepared, housing_labels)
lin_preds=lin_red.predict(housing_prepared)
tree_preds=tree_reg.predict(housing_prepared)
forest_preds=forest_reg.predict(housing_prepared)

# tree_rmses = -cross_val_score(
# tree_reg,
# housing_prepared,
# housing_labels,
# scoring="neg_root_mean_squared_error",
# cv=10
# )
# lin_rmses = -cross_val_score(
#     lin_red,
#     housing_prepared,
#     housing_labels,
#     scoring="neg_root_mean_squared_error",
#     cv=10
# )
# forest_rmses = -cross_val_score(
#     forest_reg,
#     housing_prepared,
#     housing_labels,
#     scoring="neg_root_mean_squared_error",
#     cv=10
# )

# lin_rmse = np.sqrt(mean_squared_error(housing_labels, lin_preds))
# tree_rmse = np.sqrt(mean_squared_error(housing_labels, tree_preds))
# forest_rmse = np.sqrt(mean_squared_error(housing_labels, forest_preds))


# print("Linear Regression RMSE:", lin_rmse)
# print("Decision Tree Regression RMSE:", tree_rmse)
# print("Random Forest Regression RMSE:", forest_rmse)
