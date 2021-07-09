from flask import Flask, render_template

app = Flask(__name__)


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/data-visualization')
def data_visualization():
    return render_template("data-visualization.html")


if __name__ == '__main__':
    app.run()
