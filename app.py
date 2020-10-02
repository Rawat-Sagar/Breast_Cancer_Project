from flask import Flask , render_template , request
import pickle
import numpy as np


app = Flask(__name__)

filename = 'breast_cancer_prediction_pickle_2'
model = pickle.load(open(filename,'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    radius_mean = float(request.form['radius_mean'])
    perimeter_mean  = float(request.form['perimeter_mean'])
    area_mean = float(request.form['area_mean'])
    concavity_mean = float(request.form['concavity_mean'])
    concave_points_mean = float(request.form['concave_points_mean'])
    radius_worst = float(request.form['radius_worst'])
    perimeter_worst = float(request.form['perimeter_worst'])
    area_worst = float(request.form['area_worst'])
    concavity_worst = float(request.form['concavity_worst'])
    concave_points_worst = float(request.form['concave_points_worst'])

    data = np.array([[radius_mean,perimeter_mean,area_mean ,concavity_mean,concave_points_mean,radius_worst,perimeter_worst, area_worst,concavity_worst,concave_points_worst]])
    my_prediction = model.predict(data)

    return render_template('result.html',prediction=my_prediction)

if __name__ == '__main__':
    app.run(debug=True)