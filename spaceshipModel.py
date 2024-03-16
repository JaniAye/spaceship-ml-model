import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
import pickle

# Read training and test data
train_df = pd.read_csv('./data/train.csv')
test_df = pd.read_csv('./data/test.csv')

# Fill missing values in 'Name' column
train_df['Name'] = train_df['Name'].fillna('Unknown')

# Define features and target variable
X = train_df.drop(["Transported", "PassengerId", "Name"], axis=1)  # Remove 'Name' column
y = train_df["Transported"]

# Define categorical and numerical features
categorical_features = ["HomePlanet", "CryoSleep", "Cabin", "Destination", "VIP"]
numerical_features = ["Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]

# Get indices of categorical and numerical features in X
categorical_indices = [X.columns.get_loc(col) for col in categorical_features if col in X.columns]
numerical_indices = [X.columns.get_loc(col) for col in numerical_features if col in X.columns]

# Define preprocessing steps for categorical features
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Define preprocessing steps for numerical features
numerical_transformer = SimpleImputer(strategy="mean")

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", categorical_transformer, categorical_indices),
        ("num", numerical_transformer, numerical_indices)
    ]
)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess training and testing data
X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

# Initialize and train Decision Tree model
model_dt = DecisionTreeClassifier()
model_dt.fit(X_train_transformed, y_train)

# Preprocess test data
test_df['Name'] = test_df['Name'].fillna('Unknown')
test_df_transformed = preprocessor.transform(test_df.drop(["PassengerId", "Name"], axis=1))

# Make predictions
predictions = model_dt.predict(test_df_transformed)

# Create submission DataFrame
submission_df = pd.DataFrame({"PassengerId": test_df["PassengerId"], "Transported": predictions})

# Save submission to CSV file
submission_df.to_csv("submission.csv", index=False)

# Save trained model
pickle.dump(model_dt, open("model.pkl", "wb"))
pickle.dump(preprocessor, open("preprocessor.pkl", "wb"))

print("Decision Tree model training and predictions complete!")
