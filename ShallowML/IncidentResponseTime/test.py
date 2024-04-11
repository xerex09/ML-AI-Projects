import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load data
data = pd.read_csv('incident.csv')

# Calculate time_to_resolution
data['opened_at'] = pd.to_datetime(data['opened_at'])
data['closed_at'] = pd.to_datetime(data['closed_at'])
data['time_to_resolution'] = (data['closed_at'] - data['opened_at']).dt.total_seconds() /(3600*24)  # Convert to days
 
filtered_data = data[data['time_to_resolution'] > 0]  # Remove negative values
filtered_data = filtered_data[filtered_data['time_to_resolution'] < 1e7]  # Remove outliers
filtered_data = filtered_data.dropna(subset=['assignment_group', 'contact_type'])  # Remove missing values
filtered_data = filtered_data[filtered_data['state'] == 'Closed']  # Keep only closed incidents
data = filtered_data
# Select relevant features
features = ['assignment_group', 'contact_type']  # Add Features
X = data[features]
y = data['time_to_resolution']

# Encode categorical variables if needed
X = pd.get_dummies(X)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error:", mae)