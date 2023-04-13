import numpy as np
import pandas as pd
from flask import Flask,request,render_template
from keras.models import load_model

app=Flask(__name__)
model=load_model("model.h5")

@app.route("/")
def home():
    return render_template('Main.html')

@app.route("/predict", methods=['POST'])
def predict():
    input_features =[float(x) for x in request.form.values()]
    features_values=[np.array(input_features)]
    feature_name = ['mean radius', 'mean texture', 'mean perimeter', 'mean area',
       'mean smoothness', 'mean compactness', 'mean concavity',
       'mean concave points', 'mean symmetry', 'mean fractal dimension',
       'radius error', 'texture error', 'perimeter error', 'area error',
       'smoothness error', 'compactness error', 'concavity error',
       'concave points error', 'symmetry error', 'fractal dimension error',
       'worst radius', 'worst texture', 'worst perimeter', 'worst area',
       'worst smoothness', 'worst compactness', 'worst concavity',
       'worst concave points', 'worst symmetry', 'worst fractal dimension']
    df=pd.DataFrame(features_values, columns=feature_name)
    output=model.predict(df)
    print(output)
    if output>0.5:
        return render_template("Malignant.html")
    else:
        return render_template("Benign.html")
if __name__ =="__main__":
    app.run(debug=True)