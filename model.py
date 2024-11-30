import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
import pickle
from imblearn.over_sampling import SMOTE

#PERSONALITY MODEL
#load dataset
personality_df = pd.read_csv("psyc_pycharm.csv")

#pre-processing
# Encode Gender and Target Variable
gender_encoder = LabelEncoder()
personality_df['Gender'] = gender_encoder.fit_transform(personality_df['Gender'])
personality_encoder = LabelEncoder()
personality_df['Personality'] = personality_encoder.fit_transform(personality_df['Personality'])

# Define Features and Target
X = personality_df.drop(["Personality"], axis=1)
y = personality_df["Personality"]

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply SMOTE to handle class imbalance
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X_scaled, y)

# Train the model
model = SVC(C=10, gamma=1, kernel='rbf', random_state=42)
model.fit(X_balanced, y_balanced)

#save model
with open("personality_model.pkl", "wb") as f:
    pickle.dump(model, f)
with open('gender_encoder.pkl', 'wb') as f:
    pickle.dump(gender_encoder, f)
with open('personality_encoder.pkl', 'wb') as f:
    pickle.dump(personality_encoder, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)


# X_test = ['Female',30,2,2,1,7,8]
# new_data = pd.DataFrame([X_test], columns=["Gender", "Age","openness","neuroticism","conscientiousness","agreeableness","extraversion"])
# new_data["Gender"] = gender_encoder.transform(new_data["Gender"])
# new_data_scaled = scaler.transform(new_data)  # Scale the features
#
# # Predict personality
# y_pred = model.predict(new_data_scaled)
# # Decode personality back to original labels
# predicted_personality = personality_encoder.inverse_transform(y_pred)
# # Output the result
# print(f"Predicted Personality: {predicted_personality[0]}")