from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle

app = Flask(__name__)

# Load model and scaler
regmodel = pickle.load(open('regmodel.pkl', 'rb'))
scaler = pickle.load(open('scaling.pkl', 'rb'))   # make sure this file exists

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    print(data)
    # Convert input to numpy array
    new_data = scaler.transform(np.array(list(data.values())).reshape(1, -1))
    # Make prediction
    output = regmodel.predict(new_data)
    print(output[0])
    return jsonify({'prediction': output[0]})

@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input=scaler.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output=regmodel.product(final_input)[0]
    return render_tempkate("home.html",prediction_text="The House price prediction is {}".format(output))

if __name__ == "__main__":
    app.run(debug=True)
