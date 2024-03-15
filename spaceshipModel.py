import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
import pickle

train_df =  pd.read_csv('./data/train.csv')
test_df = pd.read_csv('./data/test.csv')

test_PassengerId = test_df["PassengerId"]

categorical_features = ["HomePlanet", "CryoSleep", "Cabin", "Destination", "VIP"]
numerical_features = ["Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]

X = train_df.drop(["Transported", "PassengerId", "Name"], axis=1)
y = train_df["Transported"]

categorical_indices = [X.columns.get_loc(col) for col in categorical_features if col in X.columns]
numerical_indices = [X.columns.get_loc(col) for col in numerical_features if col in X.columns]

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

numerical_transformer = SimpleImputer(strategy="mean")

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", categorical_transformer, categorical_indices),
        ("num", numerical_transformer, numerical_indices)
    ]
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

model_dt = DecisionTreeClassifier()
model_dt.fit(X_train_transformed, y_train)

test_df_transformed = preprocessor.transform(test_df)

predictions = model_dt.predict(test_df_transformed)

submission_df = pd.DataFrame({"PassengerId": test_PassengerId, "Transported": predictions})
submission_df.to_csv("submission.csv", index=False)

print("Decision Tree model training and predictions complete!")
pickle.dump(model_dt, open("model.pkl", "wb"))