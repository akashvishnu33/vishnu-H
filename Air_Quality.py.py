
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import gradio as gr

# Load the dataset
df = pd.read_csv('air-quality-india.csv')

# Data Overview
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())
print("Duplicate Rows:", df.duplicated().sum())

# Visualizations
sns.histplot(df['PM2.5'], kde=True)
plt.title('Distribution of PM2.5')
plt.show()

sns.boxplot(data=df[['PM2.5', 'Month', 'Day']])
plt.title('Boxplot of Key Pollutants')
plt.show()

# Feature and target separation
X = df.drop('Year', axis=1)
y = df['Year']

# Encode categorical columns
categorical_cols = X.select_dtypes(include='object').columns
X[categorical_cols] = X[categorical_cols].apply(lambda col: col.astype('category').cat.codes)

# One-hot encoding
X = pd.get_dummies(X, drop_first=True)

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# Sample prediction
sample_input = [[35, 45, 21, 1, 64]]
sample_scaled = scaler.transform(sample_input)
prediction = model.predict(sample_scaled)
print("Predicted Air Quality Level:", prediction[0])

# Prepare new data input in correct format
new_data = pd.DataFrame(sample_input, columns=X.columns)
new_data_encoded = pd.get_dummies(new_data)
new_data_encoded = new_data_encoded.reindex(columns=X.columns, fill_value=0)
final_prediction = model.predict(new_data_encoded)
print("Final Air Quality Prediction:", final_prediction[0])

# Gradio interface
def predict_air_quality(PM25, Year, Month, Day):
    input_data = pd.DataFrame([[PM25, Year, Month, Day]], columns=['PM2.5', 'Year', 'Month', 'Day'])
    input_data = input_data.reindex(columns=X.columns, fill_value=0)
    input_scaled = scaler.transform(input_data)
    result = model.predict(input_scaled)
    return f"Predicted Air Quality Level: {result[0]}"

interface = gr.Interface(
    fn=predict_air_quality,
    inputs=[
        gr.Number(label="PM2.5"),
        gr.Number(label="Year"),
        gr.Number(label="Month"),
        gr.Number(label="Day")
    ],
    outputs="text",
    title="🌍 Air Quality Predictor",
    description="Predict Air Quality Level based on pollution indicators"
)

interface.launch()
