from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from joblib import load

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    request_type_str = request.method
    if request_type_str == 'GET':
        return render_template("index.html")
    else:
        temperature = request.form['temp_range']
        luminosity = request.form['lumos_range']
        radius = request.form['radius_range']
        am = request.form['am_range']
        spectral = request.form['spectral_class']

        features_array = [temperature, luminosity, radius, am, spectral]
        return render_template("prediction-result.html", prediction=features_array)


    # sample_input = np.array([[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 29560.0, 188000.000000, 6.0200, -4.010]])
    # model = load('model.joblib')
    # preds = model.predict(sample_input)
    # preds_as_str = str(preds)
    # return preds_as_str



@app.route('/data-visualization')
def data_visualization():
    return render_template("data-visualization.html")


@app.route('/prediction-result')
def prediction_result():
    return render_template("prediction-result.html")


if __name__ == '__main__':
    app.run()
