import json
import pickle

import numpy as np
import pandas as pd

from flask import Flask, request, app, jsonify, url_for, render_template


# Flask Application

app = Flask(__name__)

preprocessor = pickle.load(open("models/preprocessor.pkl", "rb"))
model = pickle.load(open("models/model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("home.html")

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    df = pd.DataFrame(data)
    print(data)
    df_scaled = pd.DataFrame(preprocessor.transform(data), columns=data.columns)
    output=model.predict(df_scaled)
    print(output[0])
    return jsonify(output[0])

@app.route('/predict',methods=['POST'])
def predict():
    data_dict = dict(request.form)
    df = pd.DataFrame([data_dict])

    df_scaled = pd.DataFrame(preprocessor.transform(df), columns=df.columns)
    output=model.predict(df_scaled)[0]
    return render_template("home.html",prediction_text="Predicted Crab Age {}".format(output))



if __name__=="__main__":
    app.run(debug=True)