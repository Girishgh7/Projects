import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Load your data
data = pd.read_csv('breast-cancer.csv')  # Replace 'your_data.csv' with your actual data file

# Data preprocessing
if 'id' in data.columns:
    data = data.drop('id', axis=1)
else:
    print("Warning: 'id' column not found in the DataFrame.")

if 'diagnosis' in data.columns:
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
else:
    print("Warning: 'diagnosis' column not found in the DataFrame.")

data = data.dropna()

# Shuffle the data and save to a new CSV file
shuffled_data = data.sample(frac=1).reset_index(drop=True)
shuffled_data.to_csv('shuffled_data.csv', index=False)

# Check for any remaining missing values
print("Missing values in each column:")
print(data.isnull().sum())

# Describe the dataset
print("Dataset description:")
print(data.describe())

# Visualize the distribution of the diagnosis
sns.countplot(x='diagnosis', data=data)
plt.title('Distribution of Diagnosis')
plt.show()

# Visualize the correlation matrix
plt.figure(figsize=(12, 10))
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

# Visualize pairplot for a subset of features
sns.pairplot(data[['diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean']], hue='diagnosis')
plt.show()

# Split the data into features and target variable
X = data.drop('diagnosis', axis=1)
y = data['diagnosis']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))

print("\nClassification Report:")
print(classification_report(y_test, predictions))

print("\nAccuracy Score:")
print(accuracy_score(y_test, predictions))

def predict_new_data(model, scaler, new_data):
    """
    Predict the diagnosis for new data using the trained model.
    
    Parameters:
    model (LogisticRegression): The trained logistic regression model.
    scaler (StandardScaler): The scaler used to standardize the data.
    new_data (DataFrame): The new data to predict.
    
    Returns:
    np.array: The predicted diagnosis (0 for benign, 1 for malignant).
    """
    # Standardize the new data
    new_data_scaled = scaler.transform(new_data)
    
    # Make predictions
    predictions = model.predict(new_data_scaled)
    print("New pred:\t",predictions)
    
    return predictions

def visualize_predictions(predictions):
    """
    Visualize the predictions.
    
    Parameters:
    predictions (np.array): The predicted diagnosis (0 for benign, 1 for malignant).
    """
    sns.countplot(x=predictions)
    plt.title('Predicted Diagnosis Distribution')
    plt.xlabel('Diagnosis')
    plt.ylabel('Count')
    plt.xticks([0, 1], ['Benign', 'Malignant'])
    plt.show()

predict_new_data(model, scaler,X_test[:7])
visualize_predictions(predictions)