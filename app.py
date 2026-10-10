import flask
from flask import Flask,render_template,request
import numpy
import pickle
import sklearn
from sklearn.linear_model import LinearRegression

with open("SLR_MODEL.pkl" , "rb") as f:
    model = pickle.load(f)


app = Flask(__name__)


@app.route("/")
def fun():
    return render_template("index.html")

@app.route("/predict",methods = ['GET','POST'])
def prediction():
    value = [float(i) for i in request.form.values()]  # [15]
    value = numpy.array(value)  # [15] -> array -> 1D
    sol = model.predict([value])[0]  # [ [15] ]
    return render_template("index.html" , prediction_text = sol)


if __name__ == "__main__":
    app.run(debug = True)