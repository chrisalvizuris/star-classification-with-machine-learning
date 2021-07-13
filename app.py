from flask import Flask, render_template
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from joblib import load

app = Flask(__name__)


@app.route('/')
def index():
    # sample_input = np.array([[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 29560.0, 188000.000000, 6.0200, -4.010]])
    # model = load('model.joblib')
    # preds = model.predict(sample_input)
    # preds_as_str = str(preds)
    # return preds_as_str
    return render_template("index.html")


@app.route('/data-visualization')
def data_visualization():
    return render_template("data-visualization.html")


if __name__ == '__main__':
    app.run()
