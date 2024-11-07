import streamlit as st
import pandas as pd
import pickle
from bs4 import BeautifulSoup
import requests
import json
import random

# Initialize session state variables
if "personality_pred" not in st.session_state:
    st.session_state["personality_pred"] = None
if "genre_input" not in st.session_state:
    st.session_state["genre_input"] = None
if "age" not in st.session_state:
    st.session_state["age"] = None
if "gender" not in st.session_state:
    st.session_state["gender"] = None
if "openness" not in st.session_state:
    st.session_state["openness"] = None
if "neuroticism" not in st.session_state:
    st.session_state["neuroticism"] = None
if "conscientiousness" not in st.session_state:
    st.session_state["conscientiousness"] = None
if "agreeableness" not in st.session_state:
    st.session_state["agreeableness"] = None
if "extraversion" not in st.session_state:
    st.session_state["extraversion"] = None
if "feedback" not in st.session_state:
    st.session_state["feedback"] = {}
if "title" not in st.session_state:
    st.session_state["title"] = []

if st.session_state["personality_pred"] is None:
    st.title("Personality Classification")
    if st.session_state["age"] is not None:
        st.write(f'Your gender: {st.session_state["gender"]}')
        st.write(f'Your age: {st.session_state["age"]}')
        st.write(f'Your openness: {st.session_state["openness"]}')
        st.write(f'Your neuroticism: {st.session_state["neuroticism"]}')
        st.write(f'Your conscientiousness: {st.session_state["conscientiousness"]}')
        st.write(f'Your agreeableness: {st.session_state["agreeableness"]}')
        st.write(f'Your extraversion: {st.session_state["extraversion"]}')
        # Model running
        if st.button("Predict"):
            with open('gender_encoder.pkl', 'rb') as f:
                gender_encoder = pickle.load(f)
            with open("personality_model.pkl", "rb") as f:
                model = pickle.load(f)
            feature_columns = ["Gender", "Age","openness","neuroticism","conscientiousness","agreeableness","extraversion"]
            column_data = [st.session_state["gender"], st.session_state["age"],st.session_state["openness"],st.session_state["neuroticism"],st.session_state["conscientiousness"],st.session_state["agreeableness"],st.session_state["extraversion"]]
            new_data = pd.DataFrame([column_data], columns=feature_columns)
            new_data['Gender'] = gender_encoder.transform(new_data['Gender'])
            personality_pred = model.predict(new_data)[0]
            st.session_state["personality_pred"] = personality_pred
            st.rerun()
    else:
        st.write('### :rainbow[**Tell us about yourself**]')
        st.write('#### Gender')
        gender = st.radio('Please select your gender.', ["Female","Male"])
        st.write('#### Age')
        age = st.number_input("How old are you?", min_value=10, max_value=60,value=30)
        st.write('#### How open are you to trying new experiences and exploring different perspectives?')
        openness = st.slider("On a scale from 1 to 8, where **1 means 'I prefer familiar routines'** and **8 means 'I actively seek out new and varied experiences'**", min_value=1, max_value=8,value=1)
        st.write('#### How likely are you to feel upset by changes or challenges in your life?')
        neuroticism = st.slider("On a scale from 1 to 8, where **1 means 'You rarely feel upset by changes or challenges.'** and **8 means 'You are quite sensitive to changes and challenges.'**", min_value=1, max_value=8,value=1)
        st.write('#### How organized and goal-oriented are you in your daily life?')
        conscientiousness = st.slider("On a scale from 1 to 8, where **1 means 'I tend to be spontaneous and go with the flow.'** and **8 means 'I am very organized and prefer to plan and stick to goals.'**", min_value=1, max_value=8,value=1)
        st.write('#### How would you describe your tendency to empathize and cooperate with others?')
        agreeableness = st.slider("On a scale from 1 to 8, where **1 means 'I tend to be more critical or assertive in my interactions.'** and **8 means 'I am highly empathetic and cooperative, often putting others' needs first.'**", min_value=1, max_value=8,value=1)
        st.write('#### How do you feel about social interactions and being around other people?')
        extraversion = st.slider("On a scale from 1 to 8, where **1 means 'I prefer solitude and quiet environments'** and **8 means 'I am energized by social gatherings and enjoy meeting new people.'**", min_value=1, max_value=8,value=1)
        if st.button('Submit'):
            st.session_state["age"] = age
            st.session_state["gender"] = gender
            st.session_state["openness"] = openness
            st.session_state["neuroticism"] = neuroticism
            st.session_state["conscientiousness"] = conscientiousness
            st.session_state["agreeableness"] = agreeableness
            st.session_state["extraversion"] = extraversion
            st.rerun()
if  st.session_state["personality_pred"] is not None and st.session_state["genre_input"] is None:
    st.title("Personality Classification")
    st.write(f"Your personality is <span style='color:red;'><b><i>{st.session_state["personality_pred"]}.", unsafe_allow_html=True)
    if st.session_state["personality_pred"] == 'extraverted':
        personality_image = 'https://i.pinimg.com/564x/36/68/78/36687829d8719ef996669db9ccb1bfb1.jpg'
    elif st.session_state["personality_pred"] == 'lively':
        personality_image = 'https://i.pinimg.com/736x/72/7b/8f/727b8f02c863018e59fc5aa8e2920b86.jpg'
    elif st.session_state["personality_pred"] == 'dependable':
        personality_image = 'https://i.pinimg.com/564x/62/a9/1f/62a91f3d7f04c77651c668c35c79c21d.jpg'
    elif st.session_state["personality_pred"] == 'responsible':
        personality_image = 'https://i.pinimg.com/564x/88/59/60/88596055c2af8cfe635bf3e12e1a30f3.jpg'
    else:
        personality_image = 'https://i.pinimg.com/564x/ad/fb/01/adfb011065094db385d615cb3b7d4afe.jpg'
    st.image(personality_image, width=300)
    if st.session_state["gender"] == 'Male':
        genre_choices = ('Khoa học viễn tưởng','Trinh thám - Kinh dị','Khoa học','Kinh doanh','Phiêu lưu ly kỳ','Kinh tế - Chính trị')
    else:
        genre_choices = ('Lãng mạn','Nghệ thuật','Tâm linh - Tôn giáo','Thơ - Kịch','Văn học hiện đại','Tâm lý học')
    genre_input = st.selectbox("You may be interested in one of these genres. Pick one:", genre_choices)
    if st.button('Submit'):
        st.session_state["genre_input"] = genre_input
        st.rerun()

book_df = pd.read_csv('book_genre.csv')
def fetch_books_data(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Connection": "keep-alive",
    }
    response = requests.get(url,headers=headers)
    data = BeautifulSoup(response.content, 'html.parser')
    img_tags = data.find_all("img", class_="lazyload img-responsive center-block")
    # data_image = img_tag.get("data-src")
    # alt_text = img_tag.get("alt")
    return img_tags
    
if st.session_state["genre_input"] is not None:
    st.title("Book Recommendation")
    url = book_df[book_df['Genre']==st.session_state["genre_input"]].iloc[0]['URLs']        
    st.write(url)
    img_tags = fetch_books_data(url)    
    random_element = random.choice(img_tags)
    st.write(f"##### <span style='color:red;'>{random_element.get("alt")}", unsafe_allow_html=True)
    st.image(random_element.get("data-src"),width=250)

# if st.session_state["genre_input"] is not None:
#     matching_books = book_df[(book_df['Main Genre'] == st.session_state["genre_input"])]
#     if not matching_books.empty:
#         first_book_url = matching_books.sample(n=1).iloc[0]['URLs']     
#         st.write(first_book_url)
#         # first_book_url = 'https://www.amazon.in/Complete-Novels-Sherlock-Holmes/dp/8175994312/ref=zg_bs_g_1318054031_d_sccl_1/000-0000000-0000000?psc=1'
#         try:
#             title, image_url, soup = fetch_open_graph_data(first_book_url)
#             st.write("### Recommended Book:")
#             st.write(f"##### <span style='color:red;'>{title}", unsafe_allow_html=True)
#             st.write(image_url)
#             # st.image(image_url, width=300)
#             # st.link_button("Buy the Book", first_book_url)
#             if st.button('Find another Book'):
#                 st.rerun()
#             # st.write("##### Do you like our recommendation?")      
#             # st.session_state["title"].append(title)
#             # sentiment_mapping = ["one", "two", "three", "four", "five"]
#             # selected = st.feedback("stars")        
#             # if selected is not None:
#             #     st.session_state["feedback"][st.session_state["title"][-2]] = sentiment_mapping[selected]
#             # st.write("##### Feedback History")      
#             # # st.write(st.session_state["feedback"])
#             # # st.write(st.session_state["title"])
#             # for title,feedback in st.session_state["feedback"].items():
#             #     if feedback:
#             #         st.write(f"You gave the book :rainbow[{title}] {feedback} stars.") 
#             #     else:
#             #         st.write(f"You haven't given feedback to the book :rainbow[{title}].")    
#             st.write(soup)                
#         except:
#             st.write('Please visit the site directly.')
#             st.write(first_book_url)    
#             title = 'not available'        
#             image_url = 'https://i.pinimg.com/736x/72/7b/8f/727b8f02c863018e59fc5aa8e2920b86.jpg'        
#             if st.button('Refresh'):
#                 st.rerun()
        
    
