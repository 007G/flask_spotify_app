from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from flask_bootstrap import Bootstrap

app = Flask(__name__)
Bootstrap(app)

pickle_in_model = open('modelForPrediction.sav', 'rb')
classifier = pickle.load(pickle_in_model)

pickle_in_scaler = open('sandardScalar.sav', 'rb')
std_scaler = pickle.load(pickle_in_scaler)


@app.route('/')
def hello_world():
    return render_template("index.html")


@app.route('/', methods=['GET', "POST"])
def predict():
  input_values = [float(x) for x in request.form.values()]
  inp_features = [input_values]
  scalar = StandardScaler()
  scaled_input = std_scaler.transform(inp_features)
  prediction = classifier.predict(scaled_input)
  if prediction==1:
    return render_template('index.html', prediction_text='Song will be listed in Billboard Hot 100 list')
  else:
    return render_template('index.html', prediction_text='Song will not be listed in Billboard Hot 100 list')


if __name__ == '__main__':
    app.run(debug=True)