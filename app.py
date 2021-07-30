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
    """
    Main function called when landing on the login page.

    :return: Template for the login page if the user is barely landing on the page.
    Else, get the email and password they entered and validate that its in login-info.csv.
    Invalid entries redirect users to invalid login page.
    """

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
    """
    Main function called when landing on the invalid login page.

    :return: Template for the login page, except an error message gets passed.
    Else, if the user inputs an email and password, it validates that those credentials
    are in the login-info.csv file.
    Invalid login attempts redirect users back to this page to try again.
    """
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
    """
    Function initiated for the main page after user logs in.

    :return: The index.html template if user is just landing on the page.
    If he user inputs information and clicks "submit", that info gets transformed into
    a numpy array and ran through the machine learning model.

    The accuracy score and predicted result gets printed on the next page.
    """
    request_type_str = request.method
    if request_type_str == 'GET':
        return render_template("index.html")
    else:
        # Create empty list that will be used to add in user's input
        user_input_list = []
        temperature = request.form['temp_range']
        luminosity = request.form['lumos_range']
        radius = request.form['radius_range']
        am = request.form['am_range']
        spectral = request.form['spectral_class']

        str_to_int(convert_spectral_to_str_floats(spectral), user_input_list)
        user_input_list.append(float(temperature))
        user_input_list.append(float(luminosity))
        user_input_list.append(float(radius))
        user_input_list.append(float(am))

        # Convert list into numpy array and run a prediction on it with model
        prediction_array = np.array([user_input_list])
        print(prediction_array)
        model = load('model.joblib')
        prediction = model.predict(prediction_array)
        preds_as_str = str(prediction)

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
        return render_template("prediction-result.html", prediction=preds_as_str, accuracy=min_as_str)


def convert_spectral_to_str_floats(user_input):
    """
    Converts spectral class into a list of floats. The transformed data will allow the input to
    run in the machine learning model.

    :param user_input: Takes in the Spectral Class from the drop down menu, which is a single letter.
    :return: Return a list of floats that correlate with the transformed values in the ML model.
    """
    if user_input == 'A':
        output = "1.0 0.0 0.0 0.0 0.0 0.0 0.0"
        return output
    elif user_input == 'B':
        output = "0.0 1.0 0.0 0.0 0.0 0.0 0.0"
        return output
    elif user_input == 'F':
        output = "0.0 0.0 1.0 0.0 0.0 0.0 0.0"
        return output
    elif user_input == 'G':
        output = "0.0 0.0 0.0 1.0 0.0 0.0 0.0"
        return output
    elif user_input == 'K':
        output = "0.0 0.0 0.0 0.0 1.0 0.0 0.0"
        return output
    elif user_input == 'M':
        output = "0.0 0.0 0.0 0.0 0.0 1.0 0.0"
        return output
    elif user_input == 'O':
        output = "0.0 0.0 0.0 0.0 0.0 0.0 1.0"
        return output


def str_to_int(spectral_class_function, emptylist):
    """
    This function converts the string output from convert_spectral_to_str_floats() into a list of floats.

    :param spectral_class_function: Takes in convert_spectral_to_str_floats() function.
    :param emptylist: Takes in the empty list that will be later converted into the numpy array.
    :return: Returns the list with the converted spectral class values appended.
    """
    int_string = spectral_class_function
    newlist = [float(i) for i in int_string.split(' ')]
    for j in newlist:
        emptylist.append(j)
    return emptylist


@app.route('/data-visualization')
def data_visualization():
    """
    Main function called when landing on the data visualization page.

    :return: The HTML template for the data visualization page. Includes graphs and a table.
    """
    return render_template("data-visualization.html")


@app.route('/prediction-result')
def prediction_result():
    """
    Main function called for the prediction result page. This is just the empty page.

    :return: HTML template for the prediction result page.
    """
    return render_template("prediction-result.html")


# run the application
if __name__ == '__main__':
    app.run()
