import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

#INTEREST MODEL
#load dataset
interest_df = pd.read_csv("interest.csv")

#pre-processing
gender_encoder = LabelEncoder()
interest_df['Gender'] = gender_encoder.fit_transform(interest_df['Gender'])
print(interest_df.head())

X = interest_df.drop(["Interest"], axis=1)
y = interest_df["Interest"]

# Train the model
model = DecisionTreeClassifier()
model.fit(X, y)

#save model
import pickle
with open("interest_model.pkl", "wb") as f:
    pickle.dump(model, f)
with open('gender_encoder.pkl', 'wb') as f:
    pickle.dump(gender_encoder, f)

#PRODUCT MODEL
#load dataset
product_df = pd.read_csv("product.csv")

#pre-processing
product_df['Gender'] = gender_encoder.fit_transform(product_df['Gender'])
interest_encoder = LabelEncoder()
product_df['Interest'] = interest_encoder.fit_transform(product_df['Interest'])
product_encoder = LabelEncoder()
product_df['Product_type'] = product_encoder.fit_transform(product_df['Product_type'])
print(product_df.head())

X = product_df.drop(["Product"], axis=1)
y = product_df["Product"]

# Train the model
model = DecisionTreeClassifier()
model.fit(X, y)

#save model
with open("product_model.pkl", "wb") as f:
    pickle.dump(model, f)
with open('interest_encoder.pkl', 'wb') as f:
    pickle.dump(interest_encoder, f)
with open('product_encoder.pkl', 'wb') as f:
    pickle.dump(product_encoder, f)