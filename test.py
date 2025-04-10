# import numpy as np
# import pandas as pd 
# import joblib
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.preprocessing import StandardScaler, LabelEncoder



# # Load dataset
# df = pd.read_csv("dataset_with_labels.csv")

# # Encode categorical labels
# label_encoder = LabelEncoder()
# df["Label"] = label_encoder.fit_transform(df["Label"])

# # Separate features and target
# X = df.drop(columns=["Label"])
# y = df["Label"]

# # Standardize features
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # Split data
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# # Train Logistic Regression model
# model = LogisticRegression()
# model.fit(X_train, y_train)

# # Save the trained model
# joblib.dump((model, scaler, label_encoder), "newmodel.pkl")

# # Load the saved model
# model, scaler, label_encoder = joblib.load("newmodel.pkl")




import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder

app = Flask(__name__)

# Load dataset
df = pd.read_csv("dataset_with_labels.csv")

# Encode categorical labels
label_encoder = LabelEncoder()
df["Label"] = label_encoder.fit_transform(df["Label"])

# Separate features and target
X = df.drop(columns=["Label"])
y = df["Label"]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save the trained model
joblib.dump((model, scaler, label_encoder), "newmodel.pkl")

# Load the saved model
model, scaler, label_encoder = joblib.load("newmodel.pkl")

@app.route('/')
def home():
    return render_template('test.html')

@app.route('/predict', methods=['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final_features = scaler.transform([np.array(int_features)])
    predict = model.predict(final_features)
    output = label_encoder.inverse_transform(predict)[0]
    return render_template('test.html', output=output)

@app.route('/results', methods=['POST'])
def results():
    data = request.get_json(force=True)
    final_features = scaler.transform([np.array(list(data.values()))])
    predict = model.predict(final_features)
    output = label_encoder.inverse_transform(predict)[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
