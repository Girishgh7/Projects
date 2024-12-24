import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# Load the data
df = pd.read_csv('Food_Delivery_Times.csv')
df.dropna(inplace=True)

# Drop the 'Order_ID' column
df = df.drop("Order_ID", axis=1)

# One-hot encode categorical variables
df = pd.get_dummies(df)

# Split the data into features and target variable
X = df.drop("Delivery_Time_min", axis=1)
y = df["Delivery_Time_min"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Function to evaluate models
def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"Model: {model.__class__.__name__}")
    print("Accuracy Score: ", accuracy_score(y_test, y_pred))
    print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred))
    print("Classification Report: \n", classification_report(y_test, y_pred))
    print("\n")

# Evaluate Logistic Regression
logistic_model = LogisticRegression(max_iter=1000)
evaluate_model(logistic_model, X_train, X_test, y_train, y_test)

# Evaluate Random Forest Classifier
rf_model = RandomForestClassifier()
evaluate_model(rf_model, X_train, X_test, y_train, y_test)

# Evaluate Decision Tree Classifier
dt_model = DecisionTreeClassifier()
evaluate_model(dt_model, X_train, X_test, y_train, y_test)

# Evaluate Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred = lr_model.predict(X_test)
print(f"Model: {lr_model.__class__.__name__}")
print("R^2 Score: ", lr_model.score(X_test, y_test))
print("\n")

# Hyperparameter tuning for Random Forest
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}
grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
print("Best parameters for Random Forest: ", grid_search.best_params_)

# Cross-validation for Logistic Regression
cv_scores = cross_val_score(logistic_model, X, y, cv=5)
print("Cross-validation scores for Logistic Regression: ", cv_scores)
print("Mean cross-validation score: ", np.mean(cv_scores))

# Save the best model
best_model = grid_search.best_estimator_
joblib.dump(best_model, 'best_model.pkl')

# Load the model and make predictions
loaded_model = joblib.load('best_model.pkl')
y_pred_loaded = loaded_model.predict(X_test)
print("Accuracy Score of loaded model: ", accuracy_score(y_test, y_pred_loaded))

# Data Visualization
sns.pairplot(df)
plt.show()
