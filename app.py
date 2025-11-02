from flask import Flask, render_template, request
import numpy as np
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# 加载模型和标准化器
loaded_model = joblib.load('logistic_regression_model.pkl')
loaded_scaler = joblib.load('standard_scaler.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # 获取用户输入
        gender = request.form['gender']
        age = int(request.form['age'])
        working_status = request.form['working_status']
        academic_pressure = float(request.form['academic_pressure']) if request.form['academic_pressure'] else np.nan
        work_pressure = float(request.form['work_pressure']) if request.form['work_pressure'] else np.nan
        cgpa = float(request.form['cgpa']) if request.form['cgpa'] else np.nan
        study_satisfaction = float(request.form['study_satisfaction']) if request.form['study_satisfaction'] else np.nan
        job_satisfaction = float(request.form['job_satisfaction']) if request.form['job_satisfaction'] else np.nan
        sleep_duration = request.form['sleep_duration']
        dietary_habits = request.form['dietary_habits']
        degree = request.form['degree']
        suicidal_thoughts = request.form['suicidal_thoughts']
        work_study_hours = int(request.form['work_study_hours'])
        financial_stress = float(request.form['financial_stress']) if request.form['financial_stress'] else np.nan
        family_history = request.form['family_history']

        # 准备输入数据
        input_data = np.array([[gender, age, working_status, academic_pressure, work_pressure, cgpa,
                              study_satisfaction, job_satisfaction, sleep_duration, dietary_habits,
                              degree, suicidal_thoughts, work_study_hours, financial_stress, family_history]])

        # 编码处理
        categorical_cols = [0, 2, 8, 9, 10, 11, 14]
        for col_index in categorical_cols:
            le = LabelEncoder()
            le.fit(['Female', 'Male'] if col_index == 0 else 
                  ['Working Professional', 'Student'] if col_index == 2 else
                  ['Less than 5 hours', '5-6 hours', 'More than 8 hours'] if col_index == 8 else
                  ['Healthy', 'Moderate', 'Unhealthy'] if col_index == 9 else
                  ['BHM', 'LLB', 'B.Pharm', 'BBA'] if col_index == 10 else
                  ['No', 'Yes'] if col_index in [11, 14] else [])
            input_data[:, col_index] = le.transform(input_data[:, col_index])

        # 处理缺失值
        input_data = np.where(pd.isnull(input_data), np.nanmean(input_data, axis=0), input_data)

        # 标准化
        input_data_scaled = loaded_scaler.transform(input_data)

        # 预测
        prediction = loaded_model.predict(input_data_scaled)[0]
        result = '低风险' if prediction == 0 else '高风险'

        return render_template('result.html', result=result)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)