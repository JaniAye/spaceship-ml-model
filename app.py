import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

flask_app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

@flask_app.route("/")
def home():
    return render_template("index.html")

@flask_app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        input_data = {
            "HomePlanet": "Earth",
            "CryoSleep": True,
            "Cabin": "G/3/S",
            "Destination": "TRAPPIST-1e",
            "VIP": False,
            "Age": 27.0,
            "RoomService": 0.0,
            "FoodCourt": 0.0,
            "ShoppingMall": 0.0,
            "Spa": 0.0,
            "VRDeck": 0.0
        }

        input_df = pd.DataFrame(input_data, index=[0])
        input_transformed = preprocess_input(input_df)

        prediction = model.predict(input_transformed)

        prediction_text = "Transported" if prediction[0] else "Not Transported"
        print(prediction_text)
        return format(prediction_text)

def preprocess_input(input_df):
    categorical_features = ["HomePlanet", "Cabin", "Destination"]
    numerical_features = ["Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]

    preprocessor = pickle.load(open("preprocessor.pkl", "rb"))

    input_transformed = preprocessor.transform(input_df)

    return input_transformed

if __name__ == "__main__":
    flask_app.run(debug=True)
