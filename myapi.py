from flask import Flask, request, render_template, jsonify
import pickle
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer


print("HELLO")

app = Flask(__name__, template_folder='./templates')

dataset_url = 'https://raw.githubusercontent.com/saha-indranil/DoctorApp/main/Symptom-severity.csv'
df1=pd.read_csv(dataset_url)
model1 = pickle.load(open('./model.pkl', 'rb'))

def pred(S1,S2,S3,S4,S5,S6,S7,S8,S9,S10,S11,S12,S13,S14,S15,S16,S17,model1):
    psymptoms = [S1,S2,S3,S4,S5,S6,S7,S8,S9,S10,S11,S12,S13,S14,S15,S16,S17]
    print(psymptoms)
    a = np.array(df1["Symptom"])
    b = np.array(df1["weight"])
    for j in range(len(psymptoms)):
        for k in range(len(a)):
            if psymptoms[j]==a[k]:
                psymptoms[j]=b[k]

    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

    psymptoms = np.array(psymptoms).reshape(1, -1)
    psymptoms_imputed = imputer.fit_transform(psymptoms)

    pred2 = model1.predict(psymptoms_imputed)
    return pred2[0]

@app.route("/")
def load_page():
    return render_template('home.html')

@app.route("/predict", methods=['POST'])
def home():
    symptoms=[]
    data = request.get_json()
    symptoms = data['symptoms']
    symptoms = symptoms + [0]*(17 - len(symptoms))
    
    disease_prediction = pred(*symptoms, model1)
    return jsonify({'disease_prediction': disease_prediction}), 200
    

if __name__=="__main__":
    app.run(debug=True)
