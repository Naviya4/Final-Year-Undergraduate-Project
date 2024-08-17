from flask import Flask, render_template, request, jsonify
import lime.lime_tabular
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import numpy as np
import joblib
import lime
import pandas as pd
from datetime import datetime
import os

app = Flask(__name__)
scaler=RobustScaler()

data=pd.read_csv('Finalized_Home_Dataset.csv')
data['house_size'] = data['house_size'].str.replace(' sqft', '').astype(float)
data['land_size'] = data['land_size'].str.replace(' perches', '').astype(float)
data['price'] = data['price'].str.replace('Rs ', '').str.replace(',', '').astype(float) / 1e6
data['price'] = np.log1p(data['price'])
data['house_size'] = np.log1p(data['house_size'])
data['land_size'] = np.log1p(data['land_size'])
label_encoder_sub_location = LabelEncoder()
data['sub_location_encoded'] = label_encoder_sub_location.fit_transform(data['sub_location'])
label_encoder_luxury_level = LabelEncoder()
data['luxury_level_encoded'] = label_encoder_luxury_level.fit_transform(data['luxury_level'])
label_encoder_garden = LabelEncoder()
data['garden_encoded'] = label_encoder_garden.fit_transform(data['garden'])
label_encoder_commercial_area = LabelEncoder()
data['commercial_area_encoded'] = label_encoder_commercial_area.fit_transform(data['commercial_area'])
s = (data.dtypes == 'object')
object_cols = list(s[s].index)
data.drop(object_cols, axis=1, inplace=True)
data.rename(columns={'sub_location_encoded': 'sub_location', 'luxury_level_encoded': 'luxury_level'}, inplace=True)
data.drop(['garden', 'commercial_area'], axis = 1, inplace=True)
data.rename(columns={'garden_encoded': 'garden', 'commercial_area_encoded': 'commercial_area'}, inplace=True)
X = data.drop(['price'], axis=1)
Y = data['price']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, test_size=0.2, random_state=42)
X_train_scaled = scaler.fit_transform(X_train.values)
X_train = pd.DataFrame(X_train_scaled, index = X_train.index, columns= X_train.columns)


model=joblib.load('voting_ensemble.pkl')
# print(type(model))


@app.route('/')
def index():
    return render_template("index.html")

def explain_model(data):
    lime_explainer=lime.lime_tabular.LimeTabularExplainer(
        X_train.values,mode='regression',feature_names=X_train.columns
    )
    lime_exp=lime_explainer.explain_instance(data.values[0],model.predict)
    lime_exp_features = lime_exp.as_list()
    explanation='<br>'
    # print(lime_exp_features)
    for feature, weight in lime_exp_features:
        if weight > 0:
            explanation += f"<b>- {feature.split()[-3].replace('_','') if '_' in feature.split()[-3] else feature.split()[-3]}" + " by {:.3f} %<br>\n </b>".format(weight*100) 
    return explanation

def predict_house_price(bedrooms, bathrooms, house_size, land_size, garden, location, commercial_area, luxury_level):
    house_size=np.log1p(house_size)
    land_size=np.log1p(land_size)

    locations=['Ahangama ', 'Akkarepattu ', 'Akuressa ', 'Alawwa ', 'Alutgama ', 'Ambalangoda ', 'Ambalantota ', 'Ampara City ', 'Ampitiya ', 'Angoda ', 'Anuradhapura City ', 'Athurugiriya ', 'Avissawella ', 'Baddegama ', 'Badulla City ', 'Balangoda ', 'Bandaragama ', 'Bandarawela ', 'Batapola ', 'Battaramulla ', 'Batticaloa City ', 'Beliatta ', 'Bentota ', 'Beruwala ', 'Bingiriya ', 'Boralesgamuwa ', 'Chilaw ', 'Colombo 1 ', 'Colombo 10 ', 'Colombo 12 ', 'Colombo 13 ', 'Colombo 14 ', 'Colombo 15 ', 'Colombo 2 ', 'Colombo 3 ', 'Colombo 4 ', 'Colombo 5 ', 'Colombo 6 ', 'Colombo 7 ', 'Colombo 8 ', 'Colombo 9 ', 'Dambulla ', 'Dankotuwa ', 'Dehiwala ', 'Delgoda ', 'Deniyaya ', 'Deraniyagala ', 'Digana ', 'Dikwella ', 'Divulapitiya ', 'Diyatalawa ', 'Eheliyagoda ', 'Elpitiya ', 'Embilipitiya ', 'Galagedara ', 'Galewela ', 'Galgamuwa ', 'Galigamuwa ', 'Galle City ', 'Gampaha City ', 'Gampola ', 'Ganemulla ', 'Gelioya ', 'Ginigathhena ', 'Giriulla ', 'Godagama ', 'Hakmana ', 'Hali Ela ', 'Hambantota City ', 'Hanwella ', 'Hatton ', 'Hikkaduwa ', 'Homagama ', 'Horana ', 'Ibbagamuwa ', 'Ingiriya ', 'Ja-Ela ', 'Jaffna City ', 'Kadawatha ', 'Kadugannawa ', 'Kaduruwela ', 'Kaduwela ', 'Kalmunai ', 'Kalutara City ', 'Kamburugamuwa ', 'Kamburupitiya ', 'Kandana ', 'Kandy City ', 'Karapitiya ', 'Kataragama ', 'Katugastota ', 'Katunayake ', 'Kegalle City ', 'Kekanadura', 'Kekirawa ', 'Kelaniya ', 'Kesbewa ', 'Kiribathgoda ', 'Kitulgala ', 'Kohuwala ', 'Kolonnawa ', 'Kottawa ', 'Kotte ', 'Kuliyapitiya ', 'Kundasale ', 'Kurunegala City ', 'Kuruwita ', 'Maharagama ', 'Malabe ', 'Marawila ', 'Matale City ', 'Matara City ', 'Matugama ', 'Mawanella ', 'Mawathagama ', 'Medawachchiya ', 'Meegoda ', 'Mihintale ', 'Minuwangoda ', 'Mirigama ', 'Monaragala City ', 'Moratuwa ', 'Mount Lavinia ', 'Nallur ', 'Narammala ', 'Nattandiya ', 'Nawala ', 'Nawalapitiya ', 'Negombo ', 'Nikaweratiya ', 'Nittambuwa ', 'Nugegoda ', 'Nuwara Eliya City ', 'Padukka ', 'Palapathwela ', 'Panadura ', 'Pannala ', 'Pannipitiya ', 'Pelmadulla ', 'Peradeniya ', 'Pilimatalawa ', 'Piliyandala ', 'Polgahawela ', 'Polonnaruwa City ', 'Puttalam City ', 'Ragama ', 'Rajagiriya ', 'Rambukkana ', 'Ratmalana ', 'Ratnapura City ', 'Rattota ', 'Ruwanwella ', 'Seeduwa ', 'Talawa ', 'Tambuttegama ', 'Tangalla ', 'Thalawathugoda', 'Tissamaharama ', 'Trincomalee City ', 'Vavuniya City ', 'Veyangoda ', 'Wadduwa ', 'Warakapola ', 'Wariyapola ', 'Wattala ', 'Wattegama ', 'Weligama ', 'Welimada ', 'Wellampitiya ', 'Wellawaya ', 'Wennappuwa ', 'Yatawatta '];
    selected_location=locations.index(location)
    selected_garden = 1 if garden == 'Yes' else 0
    selected_commercial_area= 1 if commercial_area == 'Yes' else 0
    luxury_levels=['basic', 'luxury', 'semi-luxury']
 
    selected_luxury=luxury_levels.index(luxury_level.lower())
   

    input_features = np.array([[bedrooms, bathrooms, house_size, land_size, selected_location, selected_luxury,selected_garden,  selected_commercial_area]])
    input_features=scaler.transform(input_features)
    input_features_df=pd.DataFrame(input_features, columns=X_train.columns)
    predicted_price = model.predict(input_features_df)
    model_explanation =explain_model(input_features_df)
    # print(np.expm1(predicted_price)*1e6)
    return np.expm1(predicted_price)[0],model_explanation



@app.route('/predict', methods=['POST','GET'])
def predict():
    if request.method == 'POST':
        # Get input parameters from form
        bedrooms = int(request.form['bedrooms'])
        bathrooms = int(request.form['bathrooms'])
        house_size = float(request.form['house-size'])
        land_size = float(request.form['land-size'])
        garden = request.form['garden']
        location = request.form['location']
        commercial_area = request.form['commercial-area']
        luxury_level = request.form['luxury-level']
        # print(bedrooms, bathrooms, house_size, land_size, garden, location, commercial_area, luxury_level)
        
        # Make prediction
        predicted_price,model_explanation = predict_house_price(bedrooms, bathrooms, house_size, land_size, garden, location, commercial_area, luxury_level)
        # print(feedback)
        return jsonify({'predicted_price':predicted_price,"model_explanation":model_explanation})

@app.route('/submit-feedback', methods=['POST'])
def submit_feedback():
    feedback = request.json['feedback']
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    filename = f"feedback_{timestamp}.txt"

    # Ensure feedbacks folder exists
    os.makedirs('feedbacks', exist_ok=True)

    # Save feedback to file
    with open(os.path.join('feedbacks', filename), 'w') as f:
        f.write(feedback)

    return 'Feedback saved successfully', 200   
 
if __name__ == '__main__':
    app.run(debug=True)
