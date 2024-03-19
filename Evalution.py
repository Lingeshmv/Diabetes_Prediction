import pandas as pd
import pickle
from sklearn.metrics import accuracy_score

# Load the saved model
with open('diabetes_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the test data
X_test = pd.read_csv('X_test.csv')
y_test = pd.read_csv('y_test.csv')

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model on the test set
acc = accuracy_score(y_test, y_pred)

print(y_pred)

print(f'Model accuracy on test set: {acc:.2%}')