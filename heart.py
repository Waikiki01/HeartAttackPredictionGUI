import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense

# Load your dataset
df = pd.read_excel('heart_attack_prediction_dataset.xlsx')

# Select features and target variable
features = df.drop(['Patient ID', 'Heart Attack Risk'], axis=1)
target = df['Heart Attack Risk']

# Handle categorical columns
categorical_columns = ['Sex', 'Diet', 'Country', 'Continent', 'Hemisphere']
label_encoder = LabelEncoder()

for col in categorical_columns:
    features[col] = label_encoder.fit_transform(features[col])

# Extract systolic and diastolic blood pressure from 'Blood Pressure' column
features['Systolic_BP'] = features['Blood Pressure'].apply(lambda x: int(x.split('/')[0]))
features['Diastolic_BP'] = features['Blood Pressure'].apply(lambda x: int(x.split('/')[1]))

# Drop the original 'Blood Pressure' column
features.drop('Blood Pressure', axis=1, inplace=True)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Standardize numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build a simple neural network model
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=X_train_scaled.shape[1]))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, verbose=1)



# Evaluate the model
y_pred_prob = model.predict(X_test_scaled)
y_pred = (y_pred_prob > 0.5).astype(int)
y_pred = y_pred.flatten()

# Print the results
print('Accuracy:', accuracy_score(y_test, y_pred))
print('\nConfusion Matrix:\n', confusion_matrix(y_test, y_pred))
print('\nClassification Report:\n', classification_report(y_test, y_pred))

model.save('C:/Users/Danial/OneDrive/Desktop/Miniproject/heart_model.keras')