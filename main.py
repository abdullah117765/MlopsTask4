from flask import Flask, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load dataset
df = pd.read_csv("movie_data.csv")

# Data cleaning and preprocessing
# Remove duplicates
df = df.drop_duplicates()

# Clean 'Year' column
df['Year'] = df['Year'].str.extract('(\d+)').astype(float)


df.fillna(df.mean(), inplace=True)

# Feature Engineering
# Encode categorical variables
label_encoders = {}
categorical_columns = ["Genre", "Language"]
for col in categorical_columns:
    label_encoders[col] = LabelEncoder()
    df[col] = label_encoders[col].fit_transform(df[col])

# Split data into features and target variable
X = df.drop(["ID", "Movie Name", "Rating(10)"], axis=1)
y = df["Rating(10)"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error:", mae)

# Define Flask routes
@app.route('/')
def home():
    return "Movie Rating Prediction App"

@app.route('/predict', methods=['POST'])
def predict_rating():
    data = request.get_json()
    movie_features = [data[col] for col in X.columns]
    movie_features = [movie_features]
    prediction = model.predict(movie_features)
    return jsonify({"predicted_rating": prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
