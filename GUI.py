import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from keras.models import load_model
import numpy as np

# Load the pre-trained model
model = load_model('C:/Users/Danial/OneDrive/Desktop/Miniproject/heart_model.keras')

# GUI setup
root = tk.Tk()
root.title("Heart Attack Prediction")

# GUI components
fields = ['Age', 'Sex', 'Systolic Blood Pressure', 'Diastolic Blood Pressure', 'Cholesterol',
          'Diabetes', 'Max Heart Rate', 'Exercise', 'Stress', 'Chest Pain Type']

entry_vars = {field: tk.StringVar() for field in fields}

# Entry widget for age
age_label = ttk.Label(root, text="Age:")
age_entry = ttk.Entry(root, textvariable=entry_vars['Age'])

# Place the age entry widget
age_label.grid(row=fields.index('Age'), column=0, padx=10, pady=5, sticky=tk.W)
age_entry.grid(row=fields.index('Age'), column=1, padx=10, pady=5)

# Combobox for categorical variables
sex_label = ttk.Label(root, text="Sex:")
sex_combobox = ttk.Combobox(root, textvariable=entry_vars['Sex'], values=['Male', 'Female'])

systolic_bp_label = ttk.Label(root, text="Systolic Blood Pressure:")
systolic_bp_entry = ttk.Entry(root, textvariable=entry_vars['Systolic Blood Pressure'])

diastolic_bp_label = ttk.Label(root, text="Diastolic Blood Pressure:")
diastolic_bp_entry = ttk.Entry(root, textvariable=entry_vars['Diastolic Blood Pressure'])

cholesterol_label = ttk.Label(root, text="Cholesterol:")
cholesterol_entry = ttk.Entry(root, textvariable=entry_vars['Cholesterol'])

diabetes_label = ttk.Label(root, text="Diabetes:")
diabetes_combobox = ttk.Combobox(root, textvariable=entry_vars['Diabetes'], values=['No', 'Yes'])

max_heart_rate_label = ttk.Label(root, text="Max Heart Rate:")
max_heart_rate_entry = ttk.Entry(root, textvariable=entry_vars['Max Heart Rate'])

exercise_label = ttk.Label(root, text="Exercise:")
exercise_combobox = ttk.Combobox(root, textvariable=entry_vars['Exercise'], values=['No', 'Yes'])

stress_label = ttk.Label(root, text="Stress:")
stress_entry = ttk.Entry(root, textvariable=entry_vars['Stress'])

chest_pain_label = ttk.Label(root, text="Chest Pain Type:")
chest_pain_combobox = ttk.Combobox(root, textvariable=entry_vars['Chest Pain Type'], values=['No Chest Pain', 'Mild Chest Pain', 'Moderate Chest Pain', 'Severe Chest Pain'])

# Place the widgets for categorical variables
sex_label.grid(row=fields.index('Sex'), column=0, padx=10, pady=5, sticky=tk.W)
sex_combobox.grid(row=fields.index('Sex'), column=1, padx=10, pady=5)

systolic_bp_label.grid(row=fields.index('Systolic Blood Pressure'), column=0, padx=10, pady=5, sticky=tk.W)
systolic_bp_entry.grid(row=fields.index('Systolic Blood Pressure'), column=1, padx=10, pady=5)

diastolic_bp_label.grid(row=fields.index('Diastolic Blood Pressure'), column=0, padx=10, pady=5, sticky=tk.W)
diastolic_bp_entry.grid(row=fields.index('Diastolic Blood Pressure'), column=1, padx=10, pady=5)

cholesterol_label.grid(row=fields.index('Cholesterol'), column=0, padx=10, pady=5, sticky=tk.W)
cholesterol_entry.grid(row=fields.index('Cholesterol'), column=1, padx=10, pady=5)

diabetes_label.grid(row=fields.index('Diabetes'), column=0, padx=10, pady=5, sticky=tk.W)
diabetes_combobox.grid(row=fields.index('Diabetes'), column=1, padx=10, pady=5)

max_heart_rate_label.grid(row=fields.index('Max Heart Rate'), column=0, padx=10, pady=5, sticky=tk.W)
max_heart_rate_entry.grid(row=fields.index('Max Heart Rate'), column=1, padx=10, pady=5)

exercise_label.grid(row=fields.index('Exercise'), column=0, padx=10, pady=5, sticky=tk.W)
exercise_combobox.grid(row=fields.index('Exercise'), column=1, padx=10, pady=5)

stress_label.grid(row=fields.index('Stress'), column=0, padx=10, pady=5, sticky=tk.W)
stress_entry.grid(row=fields.index('Stress'), column=1, padx=10, pady=5)

chest_pain_label.grid(row=fields.index('Chest Pain Type'), column=0, padx=10, pady=5, sticky=tk.W)
chest_pain_combobox.grid(row=fields.index('Chest Pain Type'), column=1, padx=10, pady=5)

# Function to predict heart attack risk
def predict_risk():
    try:
        # Extract input features from the GUI
        age = float(age_entry.get())
        sex = sex_combobox.get()
        systolic_bp = float(systolic_bp_entry.get())
        diastolic_bp = float(diastolic_bp_entry.get())
        cholesterol = float(cholesterol_entry.get())
        diabetes = diabetes_combobox.get()
        max_heart_rate = float(max_heart_rate_entry.get())
        exercise = exercise_combobox.get()
        stress = float(stress_entry.get())
        chest_pain_type = chest_pain_combobox.get()

        # Create a DataFrame with the input features
        input_data = pd.DataFrame({
            'Age': [age],
            'Sex': [sex],
            'Systolic Blood Pressure': [systolic_bp],
            'Diastolic Blood Pressure': [diastolic_bp],
            'Cholesterol': [cholesterol],
            'Diabetes': [diabetes],
            'Max Heart Rate': [max_heart_rate],
            'Exercise': [exercise],
            'Stress': [stress],
            'Chest Pain Type': [chest_pain_type]
        })

        # Add 15 placeholder features to match the model's input shape
        for i in range(15):
            input_data[f'Placeholder_{i}'] = 0.0

        # Handle categorical columns
        label_encoder = LabelEncoder()
        input_data['Sex'] = label_encoder.fit_transform(input_data['Sex'])
        input_data['Diabetes'] = label_encoder.fit_transform(input_data['Diabetes'])
        input_data['Exercise'] = label_encoder.fit_transform(input_data['Exercise'])
        input_data['Chest Pain Type'] = label_encoder.fit_transform(input_data['Chest Pain Type'])

        # Standardize numerical features
        scaler = StandardScaler()
        input_data_scaled = scaler.fit_transform(input_data)

        # Make a prediction
        prediction = model.predict(input_data_scaled)
        predicted_class = np.round(prediction).flatten()[0]

        # Display the prediction result
        result_label.config(text=f"Heart Attack Risk: {'High' if predicted_class == 1 else 'Low'}")

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")

# Button for prediction
predict_button = ttk.Button(root, text="Predict", command=predict_risk)
result_label = ttk.Label(root, text="Heart Attack Risk: ")

# Place the prediction button and result label
predict_button.grid(row=len(fields), column=0, columnspan=2, pady=10)
result_label.grid(row=len(fields) + 1, column=0, columnspan=2)

root.mainloop()
