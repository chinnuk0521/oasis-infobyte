import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load unemployment rate data (sample data)
data = {
    'Year': [2018, 2019, 2020, 2021],
    'Unemployment_Rate': [6.59, 4.66, 9.6, 7.2]  # Sample rates, you should replace with real data
}

df = pd.DataFrame(data)

# Exploratory Data Analysis
print(df)

# Visualization: Plotting the unemployment rate over years
plt.figure(figsize=(10, 6))
plt.plot(df['Year'], df['Unemployment_Rate'], marker='o')
plt.title('Unemployment Rate Over Years')
plt.xlabel('Year')
plt.ylabel('Unemployment Rate (%)')
plt.grid(True)
plt.show()

# Linear Regression: Predicting future unemployment rates
X = df[['Year']]
y = df['Unemployment_Rate']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

# Predicting the unemployment rate for a future year
future_year = 2030
predicted_unemployment = model.predict([[future_year]])
print(f"Predicted Unemployment Rate for {future_year}: {predicted_unemployment[0]:.2f}%")
