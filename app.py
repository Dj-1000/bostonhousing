import pickle
from flask import Flask, request,url_for, render_template, jsonify
import numpy as np
import pandas as pd

app = Flask(__name__)

regmodel = pickle.load(open('regresor.pkl','rb'))
scaler = pickle.load(open('scaler.pkl','rb'))
@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict-api', methods = ['POST'])
def predict_api():
    data = request.json['data']
    data = np.array(list(data.values())).reshape(1,-1)
    new_data = scaler.transform(data)
    output = regmodel.predict(new_data)
    print("OUTPUT :",output)
    return jsonify(output[0])

@app.route('/predict',methods = ['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    final_input = scaler.transform(np.array(data).reshape(1,-1))
    output = regmodel.predict(final_input)[0]   
    return render_template("home.html", prediction_text = f"The House price prediction is {output}.")                            
    
if __name__ == '__main__':
    app.run(debug=True)    
