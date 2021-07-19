from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
from csv import DictReader
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from joblib import load

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def login():
    request_type_str = request.method
    if request_type_str == 'GET':
        return render_template("login.html")
    else:
        email = request.form['email-input']
        password = request.form['password-input']
        with open('static/login-info.csv', 'r') as read_obj:
            csv_dict_reader = DictReader(read_obj)
            for row in csv_dict_reader:
                if email == row['Email'] and password == row['Password']:
                    return redirect(url_for('index'))
            return redirect(url_for('invalid_login'))

@app.route('/invalid-login', methods=['GET', 'POST'])
def invalid_login():
    request_type_str = request.method
    if request_type_str == 'GET':
        return render_template("login.html", validation='Email or password not found. Try again.')
    else:
        email = request.form['email-input']
        password = request.form['password-input']
        with open('static/login-info.csv', 'r') as read_obj:
            csv_dict_reader = DictReader(read_obj)
            for row in csv_dict_reader:
                if email == row['Email'] and password == row['Password']:
                    return redirect(url_for('index'))
            return redirect(url_for('invalid_login'))


@app.route('/begin-prediction', methods=['GET', 'POST'])
def index():
    request_type_str = request.method
    if request_type_str == 'GET':
        return render_template("index.html")
    else:
        # Create empty list that will be used to add in user's input
        # Begin taking in user's input
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

        # Convert list into numpy array and run a prediction on it with model
        prediction_array = np.array([empty_list])
        print(prediction_array)
        model = load('model.joblib')
        preds = model.predict(prediction_array)
        preds_as_str = str(preds)

        stars = pd.read_csv('static/stars-shuffled.csv')
        x = stars.drop(['Color', 'Spectral_Class', 'Type'], axis=1)
        y = stars['Type']
        features_train, features_test, target_train, target_test = train_test_split(x,
                                                                                    y,
                                                                                    test_size=0.2)

        model2 = RandomForestClassifier()
        model2.fit(features_train, target_train)
        cvs = cross_val_score(model2, x, y)
        minimum = min(cvs)
        min_as_percent = float(minimum) * 100.0000
        min_as_str = str(min_as_percent)
        cvs_as_str = str(cvs)
        return render_template("prediction-result.html", prediction=preds_as_str, accuracy=min_as_str)


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
