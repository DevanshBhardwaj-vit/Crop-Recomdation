from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

from model import recommend_crops

import pickle

# create a flask app

app = Flask(__name__, template_folder='templates')

# Load the model

model = pickle.load(open("model.pkl", "rb"))


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/result', methods=['POST'])
def result():
    try:
        # Collect soil profile values from the form
        nitrogen = float(request.form['nitrogen'])
        phosphorus = float(request.form['phosphorus'])
        potassium = float(request.form['potassium'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        pH = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        # Create a list of soil features
        soil_features = [nitrogen, phosphorus, potassium, temperature, humidity, pH, rainfall]

        # Get crop recommendations
        crops = recommend_crops(soil_features)

        return render_template("result.html", crops=crops, soil_features=soil_features)
    except ValueError:
        return "Invalid input. Please enter valid numbers for soil profile values."


if __name__ == '__main__':
    app.run(debug=True)
