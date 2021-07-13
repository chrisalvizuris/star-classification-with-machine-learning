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
        empty_list = []
        temperature = request.form['temp_range']
        luminosity = request.form['lumos_range']
        radius = request.form['radius_range']
        am = request.form['am_range']
        spectral = request.form['spectral_class']

        str_to_int(convert_spectral_to_str_floats(spectral), empty_list)
        empty_list.append(float(temperature))
        empty_list.append(float(luminosity))
        empty_list.append(float(radius))
        empty_list.append(float(am))

        prediction_array = np.array([empty_list])
        print(prediction_array)
        model = load('model.joblib')
        preds = model.predict(prediction_array)
        preds_as_str = str(preds)

        return render_template("prediction-result.html", prediction=preds_as_str)


def convert_spectral_to_str_floats(input):
    if input == 'A':
        output = "1.0 0.0 0.0 0.0 0.0 0.0 0.0"
        return output
    elif input == 'B':
        output = "0.0 1.0 0.0 0.0 0.0 0.0 0.0"
        return output
    elif input == 'F':
        output = "0.0 0.0 1.0 0.0 0.0 0.0 0.0"
        return output
    elif input == 'G':
        output = "0.0 0.0 0.0 1.0 0.0 0.0 0.0"
        return output
    elif input == 'K':
        output = "0.0 0.0 0.0 0.0 1.0 0.0 0.0"
        return output
    elif input == 'M':
        output = "0.0 0.0 0.0 0.0 0.0 1.0 0.0"
        return output
    elif input == 'O':
        output = "0.0 0.0 0.0 0.0 0.0 0.0 1.0"
        return output


def str_to_int(input, emptylist):
    int_string = input
    newlist = [float(i) for i in int_string.split(' ')]
    for j in newlist:
        emptylist.append(j)
    return emptylist


@app.route('/data-visualization')
def data_visualization():
    return render_template("data-visualization.html")


@app.route('/prediction-result')
def prediction_result():
    return render_template("prediction-result.html")


if __name__ == '__main__':
    app.run()
