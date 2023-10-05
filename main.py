# Regression with crab age dataset


# Libraries
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression, RANSACRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import pickle as pkl


# Importing csv's
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

# Dropping id's
train_df.drop(['id'], axis=1, inplace=True)

train_df = pd.get_dummies(train_df, columns=['Sex'])


cont_features = ['Length', 'Diameter', 'Height', 'Weight', 'Shucked Weight', 'Viscera Weight', 'Shell Weight']
pipeline = Pipeline([
    ('scaler', StandardScaler())
])

processor = ColumnTransformer([
    ('cont_pipeline', pipeline, cont_features)
], remainder="passthrough")

# Train Test Split
X = train_df.drop(['Age'], axis=1)
Y = train_df['Age']
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

X_train = pd.DataFrame(processor.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(processor.transform(X_val), columns=X_val.columns)


with open("models/preprocessor.pkl", "wb") as p:
    # Dump the model object to the file
    pkl.dump(processor, p)


# Models 
models = []
model_r2 = []
model_mae = []
model_mse = []

# Models to be used
lin_reg = LinearRegression()
ransac = RANSACRegressor(
    LinearRegression(),
    max_trials=4,
	min_samples=2,
	loss='absolute_error',
	residual_threshold=10
)
tree = DecisionTreeRegressor(max_depth=5)
forest = RandomForestRegressor(
    n_estimators=20,
    max_depth=10,
    criterion="squared_error"
)

# Evaluation Metrics
def print_scores(Y_true, Y_pred, model):
    models.append(model)
    r2 = r2_score(Y_true, Y_pred)
    sq = mean_squared_error(Y_true, Y_pred)
    absl = mean_absolute_error(Y_true, Y_pred)
    model_r2.append(r2)
    model_mae.append(sq)
    model_mse.append(absl)
    print(f"R2 Score : {r2}")
    print(f"Mean Squared Error : {sq}")
    print(f"Mean Absolute Error : {absl}")


# RandomForest Regressor
forest.fit(X_train, Y_train)
Y_pred_forest = forest.predict(X_val)
print_scores(Y_val, Y_pred_forest, "Random Forest Regression")


with open("models/model.pkl", "wb") as f:
    # Dump the model object to the file
    pkl.dump(forest, f)