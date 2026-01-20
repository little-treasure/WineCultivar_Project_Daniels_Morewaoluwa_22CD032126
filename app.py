from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load trained model and scaler
model = joblib.load('model/wine_cultivar_model.pkl')
scaler = joblib.load('model/scaler.pkl')

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None

    if request.method == 'POST':
        features = [
            float(request.form['alcohol']),
            float(request.form['malic_acid']),
            float(request.form['ash']),
            float(request.form['flavanoids']),
            float(request.form['color_intensity']),
            float(request.form['proline'])
        ]

        scaled_features = scaler.transform([features])
        result = model.predict(scaled_features)[0]

        prediction = f"Cultivar {result + 1}"

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
