import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import pickle

#PERSONALITY MODEL
#load dataset
personality_df = pd.read_csv("personality_train.csv")

#pre-processing
gender_encoder = LabelEncoder()
personality_df['Gender'] = gender_encoder.fit_transform(personality_df['Gender'])

X = personality_df.drop(["Personality"], axis=1)
y = personality_df["Personality"]

# Train the model
model = KNeighborsClassifier()
model.fit(X, y)
X_test = [1,30,2,2,1,7,8]
new_data = pd.DataFrame([X_test], columns=["Gender", "Age","openness","neuroticism","conscientiousness","agreeableness","extraversion"])
y_pred = model.predict(new_data)
print(y_pred)

#save model
with open("personality_model.pkl", "wb") as f:
    pickle.dump(model, f)
with open('gender_encoder.pkl', 'wb') as f:
    pickle.dump(gender_encoder, f)
