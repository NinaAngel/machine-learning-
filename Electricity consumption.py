
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
#we Create a DataFrame from the provided data
data = {
    'Date': ['2023-11-03'] * 24,
    'Time': [
        '00:00', '01:00', '02:00', '03:00', '04:00', '05:00', '06:00', '07:00', '08:00', '09:00', '10:00', '11:00',
        '12:00', '13:00', '14:00', '15:00', '16:00', '17:00', '18:00', '19:00', '20:00', '21:00', '22:00', '23:00'
    ],
    'Location': ['Kampala'] * 24,
    'Electricity Consumption': [
        500, 450, 400, 350, 300, 250, 200, 150, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700,
        650, 600, 550
    ],
    'Temperature': [25, 24, 23, 22, 21, 20, 19, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 29, 28, 27, 26]
}

df = pd.DataFrame(data)

# Convert 'Date' and 'Time' columns to a single datetime column
df['Timestamp'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])

# Drop the 'Date' and 'Time' columns, as we now have a 'Timestamp' column and set it as the index
df = df.drop(['Date', 'Time'], axis=1)

df.set_index('Timestamp', inplace=True)




# Split the data into features (temp) and target (elec consu)
X = df['Temperature'].values.reshape(-1, 1)
y = df['Electricity Consumption'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train a regression model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate the R-squared score (model accuracy)
r2 = r2_score(y_test, y_pred)
accuracy_percentage = r2 * 100

# Visualize the electricity usage pattern
plt.figure(figsize=(12, 6))
plt.scatter(X_test, y_test, color='b', label='Actual')
plt.plot(X_test, y_pred, color='r', linewidth=2, label='Predicted')
plt.xlabel('Temperature')
plt.ylabel('Electricity Consumption')
plt.title(f'Electricity Usage Pattern Prediction\nAccuracy (R-squared): {accuracy_percentage:.2f}%')
plt.legend()
plt.show()

# Print the accuracy as a percentage
print(f"Model Accuracy (R-squared): {accuracy_percentage:.2f}%")



