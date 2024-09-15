import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import seaborn as sns

# Load the dataset
file_path = 'path_to_your_downloaded_csv_file.csv'
data = pd.read_csv(file_path)

# Display the first few rows
print(data.head())
# Check for missing values
print(data.isnull().sum())

# Fill or drop missing values if necessary
data = data.fillna(0)  # Replace NaNs with 0

# Feature engineering: Convert date column to datetime if applicable
data['date'] = pd.to_datetime(data['date'])
data['month'] = data['date'].dt.month
data['day'] = data['date'].dt.day
data['dayofweek'] = data['date'].dt.dayofweek
data['year'] = data['date'].dt.year

# Drop unnecessary columns
# Example: Drop 'Page' column if it exists (depends on the dataset)
data = data.drop(columns=['Page'], axis=1)
# Plot web traffic over time
plt.figure(figsize=(10, 6))
plt.plot(data['date'], data['traffic'], label='Traffic')
plt.title('Web Traffic Over Time')
plt.xlabel('Date')
plt.ylabel('Traffic')
plt.legend()
plt.show()

# Plot correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()
# Define features (X) and target variable (y)
X = data.drop(columns=['traffic', 'date'], axis=1)
y = data['traffic']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f'Root Mean Squared Error: {rmse}')

# Plot actual vs predicted traffic
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='Actual Traffic')
plt.plot(y_pred, label='Predicted Traffic')
plt.title('Actual vs Predicted Web Traffic')
plt.xlabel('Test Samples')
plt.ylabel('Traffic')
plt.legend()
plt.show()
# Combine predictions with test data
results = X_test.copy()
results['Actual_Traffic'] = y_test
results['Predicted_Traffic'] = y_pred

# Export to CSV
results.to_csv('web_traffic_predictions.csv', index=False)
