import streamlit as st
import pandas as pd
import pickle

# Initialize session state variables
if "product_pred" not in st.session_state:
    st.session_state["product_pred"] = None
if "interest_pred" not in st.session_state:
    st.session_state["interest_pred"] = None
if "interest_input" not in st.session_state:
    st.session_state["interest_input"] = None
if "product_type" not in st.session_state:
    st.session_state["product_type"] = None
if "age" not in st.session_state:
    st.session_state["age"] = None
if "gender" not in st.session_state:
    st.session_state["gender"] = None

if st.session_state["interest_pred"] is None:
    st.title("Interest Classification")
    if st.session_state["age"] is not None:
        st.write(f'Your age: {st.session_state["age"]}')
        st.write(f'Your gender: {st.session_state["gender"]}')
        if st.button("Predict"):
            with open('gender_encoder.pkl', 'rb') as f:
                gender_encoder = pickle.load(f)
            with open("interest_model.pkl", "rb") as f:
                model = pickle.load(f)
            feature_columns = ["Age", "Gender"]
            new_data = pd.DataFrame([[st.session_state["age"], st.session_state["gender"]]], columns=feature_columns)
            new_data['Gender'] = gender_encoder.transform(new_data['Gender'])
            interest_pred = model.predict(new_data)[0]
            st.session_state["interest_pred"] = interest_pred
            st.rerun()
    else:
        age = st.number_input("Age", min_value=10, max_value=30)
        gender = st.selectbox("Gender", ("Male", "Female"))
        if st.button('Submit'):
            st.session_state["age"] = age
            st.session_state["gender"] = gender
            st.rerun()
if  st.session_state["interest_pred"] is not None and st.session_state["interest_input"] is None:
    st.title("Interest Classification")
    st.write(f"Many of your peers are interested in {st.session_state["interest_pred"]}.")
    interest_input = st.selectbox("What do you want to explore?", ("Education", "Game", "Travel", "Fashion"))
    if st.button('Submit'):
        st.session_state["interest_input"] = interest_input
        st.rerun()
if st.session_state["interest_input"] is not None and st.session_state["product_type"] is None:
    st.title("Product Classification")
    if st.session_state["interest_input"] == 'Fashion':
        st.write(f"Great! Let's explore more about {st.session_state['interest_input']}.")
        product_type = st.selectbox("Occasion", ("Formal", "Informal"))
    elif st.session_state["interest_input"] == 'Game':
        st.write(f"Great! Let's explore more about {st.session_state['interest_input']}.")
        product_type = st.selectbox("Game type", ("PC", "Mobile"))
    elif st.session_state["interest_input"] == 'Education':
        st.write(f"Great! Let's explore more about {st.session_state['interest_input']}.")
        product_type = st.selectbox("Purpose", ("Schooling", "Leisure"))
    elif st.session_state["interest_input"] == 'Travel':
        st.write(f"Great! Let's explore more about {st.session_state['interest_input']}.")
        product_type = st.selectbox("Item type", ("Luggage", "Accessories"))
    if st.button('Submit'):
        st.session_state["product_type"] = product_type
        st.rerun()

if st.session_state['product_type'] is not None and st.session_state["product_pred"] is None:
    st.title(f"{st.session_state['interest_input']} Classification")
    st.write(f'You have picked: {st.session_state["product_type"]}')
    if st.button("Shop"):
        with open('gender_encoder.pkl', 'rb') as f:
            gender_encoder = pickle.load(f)
        with open('interest_encoder.pkl', 'rb') as f:
            interest_encoder = pickle.load(f)
        with open('product_encoder.pkl', 'rb') as f:
            product_encoder = pickle.load(f)
        with open("product_model.pkl", "rb") as f:
            model = pickle.load(f)
        feature_columns = ["Age", "Gender","Interest","Product_type"]
        age = st.session_state['age']
        gender = st.session_state['gender']
        interest = st.session_state['interest_input']
        product_type = st.session_state['product_type']
        new_data = pd.DataFrame([[age, gender,interest,product_type]], columns=feature_columns)
        # new_data = pd.DataFrame([[10,'Male','Education','Schooling']], columns=feature_columns)
        new_data['Gender'] = gender_encoder.transform(new_data['Gender'])
        new_data['Interest'] = interest_encoder.transform(new_data['Interest'])
        new_data['Product_type'] = product_encoder.transform(new_data['Product_type'])
        product_pred = model.predict(new_data)[0]
        st.session_state["product_pred"] = product_pred
        st.rerun()

if st.session_state["product_pred"] is not None:
    st.title("Product Classification")
    st.write(f'You should buy {st.session_state["product_pred"]} product.')

# st.write('# Sentiment Analysis App')
# pipe1 = pipeline("sentiment-analysis")
# user_text = prediction
# if st.button("Analyze"):
#     result = pipe1(user_text)
#     st.write(f"Sentiment: {result[0]['label']}")
#     if result[0]['label'] == 'NEGATIVE':
#         tts.text_to_speech('I am feeling very happy.')
#     else:
#         tts.text_to_speech('I am feeling sad.')
#     st.audio("speech.wav", format="wav",autoplay=True)


