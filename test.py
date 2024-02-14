import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the trained model
model = load_model('C:/Users/Danial/OneDrive/Desktop/Miniproject/heart_model.keras')

# Load new data that you want to make predictions on
new_data = pd.read_excel('C:/Users/Danial/OneDrive/Desktop/Miniproject/newdata.xlsx')

# Ensure the same preprocessing steps as during training

# Handle categorical columns


# Extract systolic and diastolic blood pressure from 'Blood Pressure' column
new_data['Systolic_BP'] = new_data['Blood Pressure'].apply(lambda x: int(x.split('/')[0]))
new_data['Diastolic_BP'] = new_data['Blood Pressure'].apply(lambda x: int(x.split('/')[1]))

# Drop the original 'Blood Pressure' column
new_data.drop('Blood Pressure', axis=1, inplace=True)

# Ensure numerical features are on the same scale as during training
scaler = StandardScaler()
new_data_scaled = scaler.fit_transform(new_data)

# Make predictions
predictions_proba = model.predict(new_data_scaled)
predictions = (predictions_proba > 0.5).astype(int)
predictions = predictions.flatten()

# Print the predictions
print('Heart Attack: ',  predictions)
