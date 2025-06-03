import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Load the dataset
data = pd.read_csv('weatherAUS.csv')

# Define numerical and categorical columns
numerical_cols = ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine',
                  'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am',
                  'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am',
                  'Cloud3pm', 'Temp9am', 'Temp3pm']
categorical_cols = ['WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday', 'RainTomorrow']

# Impute numerical columns with mean
num_imputer = SimpleImputer(strategy='mean')
data[numerical_cols] = num_imputer.fit_transform(data[numerical_cols])

# Impute categorical columns with the most frequent value
cat_imputer = SimpleImputer(strategy='most_frequent')
data[categorical_cols] = cat_imputer.fit_transform(data[categorical_cols])

# Create Temperature Range feature
data['TempRange'] = data['MaxTemp'] - data['MinTemp']


# Convert back to DataFrame
features = pd.DataFrame(data, columns=numerical_cols)



from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
# Apply Isolation Forest
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)
iso_forest = IsolationForest(contamination=0.01, random_state=42)
features['iso_forest_anomaly'] = iso_forest.fit_predict(X_scaled)

# Step 4: Apply Local Outlier Factor
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.01)
features['lof_anomaly'] = lof.fit_predict(X_scaled)

# Step 5: Ensemble - mark as anomaly only if both methods agree
features['final_anomaly'] = ((features['iso_forest_anomaly'] == -1) & (features['lof_anomaly'] == -1)).astype(int)

# Step 6: Filter out anomalies to get cleaned data
cleaned_features = features[features['final_anomaly'] == 0].reset_index(drop=True)

print(f"Original size after manual filtering: {len(features)}")
print(f"Cleaned size after anomaly removal: {len(cleaned_features)}")

anomalies = features[features['final_anomaly'] == 1]

print(len(anomalies))
print(anomalies.head())

clean_indices = features[features['final_anomaly'] == 0].index
clean_data = data.loc[clean_indices].reset_index(drop=True)
