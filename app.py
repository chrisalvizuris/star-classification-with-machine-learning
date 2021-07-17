from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from csv import writer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
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
        # Create empty list that will be used to add in user's input
        # Begin taking in user's input
        empty_list = []
        # original_list = []
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
        #
        # original_list.append(int(temperature))
        # original_list.append(float(luminosity))
        # original_list.append(float(radius))
        # original_list.append(float(am))
        # original_list.append("Red")
        # original_list.append(spectral)
        # original_list.append(int(preds))
        # with open('static/stars-shuffled.csv', 'a') as f_object:
        #     writer_object = writer(f_object)
        #     writer_object.writerow(original_list)
        #     f_object.close()

        stars = pd.read_csv('static/stars-shuffled.csv')
        x = stars.drop(['Color', 'Spectral_Class', 'Type'], axis=1)
        y = stars['Type']

        # categorical_features = ["Spectral_Class"]
        # one_hot = OneHotEncoder()
        # transformer = ColumnTransformer([("one_hot",
        #                                   one_hot,
        #                                   categorical_features)],
        #                                 remainder="passthrough")
        #
        # transformed_features = transformer.fit_transform(x)
        #
        # features_train, features_test, target_train, target_test = train_test_split(transformed_features,
        #                                                                             y,
        #                                                                             test_size=0.2)
        features_train, features_test, target_train, target_test = train_test_split(x,
                                                                                    y,
                                                                                    test_size=0.2)

        model2 = RandomForestClassifier()
        model2.fit(features_train, target_train)
        cvs = cross_val_score(model2, x, y)
        cvs_as_str = str(cvs)
        return render_template("prediction-result.html", prediction=preds_as_str, accuracy=cvs_as_str)


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
