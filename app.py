from flask import Flask, render_template, request
from model import predict_cgpa

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    gender = int(request.form['gender'])
    parental_level_of_education = int(request.form['parental_level_of_education'])
    lunch = int(request.form['lunch'])
    test_preparation = int(request.form['test_preparation'])
    hours_req_per_unit = float(request.form['hours_req_per_unit'])
    hours_of_sleep = float(request.form['hours_of_sleep'])
    lastsemcgpa = float(request.form['lastsemcgpa'])
    cgpabefore = float(request.form['cgpabefore'])

    input_data = {
        'gender': [gender],
        'parental_level_of_education': [parental_level_of_education],
        'lunch': [lunch],
        'test_preparation': [test_preparation],
        'hours_req_per_unit': [hours_req_per_unit],
        'hours_of_sleep': [hours_of_sleep],
        'lastsemcgpa': [lastsemcgpa],
        'cgpabefore': [cgpabefore]
    }

    predicted_cgpas = predict_cgpa(input_data)
    return render_template('index.html', predictions=predicted_cgpas)

if __name__ == '__main__':
    app.run(debug=True)
