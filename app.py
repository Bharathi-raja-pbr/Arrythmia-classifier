from flask import Flask, request, render_template, send_file, flash, redirect, url_for
from tensorflow.keras.models import load_model 
import webview
import os
import numpy as np
import pandas as pd
import  pywt
from scipy import stats
import matplotlib.pyplot as plt

# Load the trained model
model = load_model("/Users/annamalaiiyappan/Desktop/FYP/env/model.h5")

app = Flask(__name__, template_folder='templates')


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/converter', methods=['POST','GET'])
def converter():
    if request.method == 'POST':
        file = request.files['file']
        if file.filename.endswith('.dat'):
            # Load the .dat file
            data = np.fromfile(file, dtype=np.int16)

            # Create a DataFrame
            df = pd.DataFrame({"ECG_Data": data})

            # Save the DataFrame to a CSV file
            output_file = "output.csv"
            df.to_csv(output_file, index=False)

            # Provide a link to download the converted file
            return send_file(output_file, as_attachment=True)
        else:
            return 'Invalid file format. Please upload a .dat file.'
    return render_template('converter.html')


@app.route('/download/<filename>')
def download(filename):
    # Check if the file exists
    if os.path.exists(filename):
        # Provide a link to download the file
        return send_file(filename, as_attachment=True)
    else:
        return 'File not found.'


@app.route('/prediction')
def prediction():
    return render_template('prediction.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Get the uploaded CSV file
    file = request.files['file']
    # Read the CSV file
    data = pd.read_csv(file)

    # Step 2: Extract the features
    X = data.iloc[:, :-1]  # Features are all columns except the last one

    # Step 3: Convert to NumPy array
    X_array = X.values

    # Step 4: Reshape the feature array
    num_features = 360
    X_array = np.pad(X_array, ((0, 0), (0, num_features - X_array.shape[1])), mode='constant')
    X_array = X_array.reshape(len(X_array), num_features, 1)

    # Make predictions
    y_pred = model.predict(X_array)

    # Convert predictions to class labels
    y_pred_val = np.argmax(y_pred, axis=1)[0]

    classes=["Normal Sinus Rythm", "  Left bundle branch block"," Right bundle branch block" ," Atrial premature beat","Premature ventricular contraction (PVC)"]
    class_details = [
   "In a normal sinus rhythm, the heart's electrical signals originate from the sinus node (the heart's natural pacemaker) and follow a regular pattern. This rhythm is considered normal and indicates that the heart is functioning correctly.",
  "The heart's electrical system includes the left and right bundle branches, which help to coordinate the contraction of the heart muscle. A blockage or delay in the left bundle branch can disrupt the normal electrical signals, leading to an abnormal heart rhythm.",
  "A right bundle branch block is a condition where there is a delay or blockage in the electrical signals that control the contraction of the right ventricle, one of the lower chambers of the heart. This delay can cause the right ventricle to contract later than normal, which can lead to an abnormal heart rhythm. Right bundle branch block can be caused by various factors, including heart disease, high blood pressure, or certain medications.",
  "A premature beat occurs when the atria (the upper chambers of the heart) contract earlier than they should. This can disrupt the normal heart rhythm and is often felt as a 'skipped' or 'extra' beat.",
    "Premature ventricular contractions occur when the ventricles (the lower chambers of the heart) contract earlier than they should. This can also disrupt the normal heart rhythm and is often felt as a 'skipped' or 'extra' beat."
]

    # Redirect to the result page with the prediction and description
    print("result done")
    return redirect(url_for('result', prediction=classes[y_pred_val], des=class_details[y_pred_val]))


@app.route('/result')
def result():
    print("result opened")
    prediction = request.args.get('prediction')
    des = request.args.get('des')
    print("result displayed")
    return render_template('result.html', prediction=prediction, des=des)
    


if __name__ == '__main__':
     app.run(debug=False,host='0.0.0.0')
