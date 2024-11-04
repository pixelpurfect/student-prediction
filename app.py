from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder
import openai

app = Flask(__name__)

# Set your OpenAI API Key
openai.api_key = 'key'  # Replace with your actual API key

# Load the dataset
data = pd.read_csv('studs.csv')

# Function to assign study hours and sleep hours based on CGPA
def assign_hours(cgpa, max_cgpa, min_cgpa):
    normalized_cgpa = (cgpa - min_cgpa) / (max_cgpa - min_cgpa)
    hours_req_per_unit = round((1 - normalized_cgpa) * 11 + 1)
    sleep_hours = round(normalized_cgpa * 7 + 1)
    return hours_req_per_unit, sleep_hours

# Prepare the data
max_cgpa = data['cgpa'].max()
min_cgpa = data['cgpa'].min()
data[['hours_req_per_unit', 'hours_of_sleep']] = data['cgpa'].apply(
    lambda x: assign_hours(x, max_cgpa, min_cgpa)).apply(pd.Series)

# Encode categorical variables
label_encoders = {}
for column in ['gender', 'parental_level_of_education', 'lunch', 'test_preparation']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Define features and target variable
X = data[['gender', 'parental_level_of_education', 'lunch', 'test_preparation', 'hours_req_per_unit', 'hours_of_sleep', 'lastsemcgpa', 'cgpabefore']]
y = data['cgpa']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the Ridge Regression model
ridge_model = Ridge()
ridge_model.fit(X_train, y_train)

# Function to predict CGPA using Ridge Regression
def predict_cgpa(input_data):
    input_df = pd.DataFrame(input_data)
    pred = ridge_model.predict(input_df)
    return pred[0]

# Function to generate suggestions using OpenAI API based on provided input
def generate_suggestions(cgpa, hours_req_per_unit, sleep_hours):
    prompt = (
        f"Based on the following student details:\n"
        f"CGPA: {cgpa}\n"
        f"Hours required to complete a unit: {hours_req_per_unit} hours\n"
        f"Hours of sleep: {sleep_hours} hours a day\n"
        f"Please provide personalized suggestions to improve academic performance. "
        f"1. If the CGPA is 10, appreciate the student's effort and suggest maintaining high performance through advanced coursework or leadership roles in study groups.\n"
        f"2. If the CGPA is between 7 and 10, acknowledge their achievements while encouraging them to set higher goals and explore opportunities for enrichment in their studies.\n"
        f"3. If the CGPA is between 5 and 7, encourage consistent study habits, recommend utilizing available academic resources like tutoring, and emphasize the importance of time management.\n"
        f"4. If the CGPA is below 5, provide constructive feedback on the necessity of developing effective study strategies, suggest reaching out for academic support, and emphasize the importance of dedication and effort in their studies."
    )

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150
    )

    return response['choices'][0]['message']['content']

# Function to generate a personalized study plan
def generate_study_plan(cgpa, subjects, study_hours_available, exam_date):
    prompt = (
        f"Create a personalized study plan for a student with a CGPA of {cgpa}.\n"
        f"They have the following subjects to study: {', '.join(subjects)}.\n"
        f"They have {study_hours_available} hours available for study each day.\n"
        f"The exam date is {exam_date}.\n"
        f"Please provide a detailed weekly study schedule, including daily topics and recommended resources."
    )

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300
    )

    return response['choices'][0]['message']['content']

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get user input from the form
        user_cgpa = float(request.form['previous_cgpa'])  # Use CGPA entered by the user for suggestions
        input_data = {
            'gender': [int(request.form['gender'])],
            'parental_level_of_education': [int(request.form['education'])],
            'lunch': [int(request.form['lunch'])],
            'test_preparation': [int(request.form['test_preparation'])],
            'hours_req_per_unit': [float(request.form['study_hours'])],
            'hours_of_sleep': [float(request.form['sleep_hours'])],
            'lastsemcgpa': [float(request.form['last_sem_cgpa'])],
            'cgpabefore': [user_cgpa]
        }
        
        predicted_cgpa = predict_cgpa(input_data)

        # Calculate hours required and sleep hours based on user input CGPA
        hours_req_per_unit, sleep_hours = assign_hours(user_cgpa, max_cgpa, min_cgpa)

        # Generate suggestions using the user-provided CGPA
        suggestions = generate_suggestions(user_cgpa, hours_req_per_unit, sleep_hours)

        # Example study plan inputs
        subjects = ["Formal Language and Automata", "Computer Networks", "Machine Learning"]
        study_hours_available = float(request.form['study_hours'])
        exam_date = "2024-11-08"  # Modify as needed
        study_plan = generate_study_plan(predicted_cgpa, subjects, study_hours_available, exam_date)

        return render_template('index.html', predicted_cgpa=predicted_cgpa, suggestions=suggestions, study_plan=study_plan)
    
    return render_template('index.html', predicted_cgpa=None, suggestions=None, study_plan=None)

if __name__ == '__main__':
    app.run(debug=True)
