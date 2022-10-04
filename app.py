import numpy as np
from flask import Flask, jsonify, request, render_template

import pickle

app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))
names = pickle.load(open("iris_names.pkl", "rb"))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    features = [float(x) for x in request.form.values()]
    print(features)
    final_features = [np.array(features)]
    pred = model.predict(final_features)

    print (pred[0] == 0 and "Feminino" or "Masculino")

    


    output = str(pred[0] == 0 and "Masculino" or "Feminino")

    

    return render_template("index.html", prediction_text="Resultado: " + output)


@app.route("/api", methods=["POST"])
def results():
    data = request.get_json(force=True)
    pred = model.predict([np.array(list(data.values()))])

    output = names[pred[0]]
    return jsonify(output)
