import numpy as np
import pickle
from flask import Flask, render_template, request, jsonify



# create flask app
app = Flask(__name__)

# load pickle models
uti = pickle.load(open('uti.pkl', 'rb'))
bsi = pickle.load(open('bsi.pkl', 'rb'))


@app.route("/")
def main():
    return render_template("hello.html")

@app.route("/predict", methods=["POST"])
def home():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    y1 = uti.predict(features)
    y2 = "Positive" if y1 == 1 else y1 == "Negative"
    y3 = str(round(np.max(uti.predict_proba(features)) * 100, 2))
    z1 = bsi.predict(features)
    z2 = "Positive" if z1 == 1 else z1 == "Negative"
    z3 = str(round(np.max(bsi.predict_proba(features)) * 100, 2))

    a1 = "UTI :" + str(y2) + " (" + str(y3) + "%)" + " / Urinary tract-related BSI: " +  " (" + str(z3) + "%)"

    return render_template("index.html", prediction_uti='{}'.format(a1))

if __name__ == "__main__":
    app.run(debug=False)
