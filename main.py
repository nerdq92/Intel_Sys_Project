import streamlit as st
import pandas as pd
import pickle
import random
import csv
import time
from API_functions import fetch_books_data,fetch_books_description

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
if "title" not in st.session_state:
    st.session_state["title"] = []
if "genre_choices" not in st.session_state:
    st.session_state["genre_choices"] = ()
if "random_choice" not in st.session_state:
    st.session_state["random_choice"] = None
if "toast" not in st.session_state:
    st.session_state["toast"] = None

# logo display
left,middle,right=st.columns(3)
middle.image('bookwise_logo.png',use_column_width=True)

# personality questionnaire
if st.session_state["personality_pred"] is None:
    st.markdown(
        """
        <h2 style="text-align: center; color: #d86cf5;">Personality Classification</h2>
        """,
        unsafe_allow_html=True
    )
    # after completed questionnaire
    if st.session_state["age"] is not None:
        # display questionnaire information to user for confirmation
        st.balloons()
        st.write(f'Your gender: :rainbow[{st.session_state["gender"]}]')
        st.write(f'Your age: :rainbow[{st.session_state["age"]}]')
        st.write('Your scores:')
        row_data = [['openness',st.session_state["openness"]],['neuroticism',st.session_state["neuroticism"]],['conscientiousness',st.session_state["conscientiousness"]],['agreeableness',st.session_state["agreeableness"]],['extraversion',st.session_state["extraversion"]]]
        chart_data = pd.DataFrame(row_data,columns=['category','score'])
        st.bar_chart(chart_data,x='category',y='score')
        if st.session_state["toast"] == None:
            st.toast("Thank you for filling out the questionaire.",icon='✨')
            time.sleep(1)
            st.toast("You're doing great!", icon='🎉')
            time.sleep(1)
            st.toast("Don't forget to confirm your information and hit Predict!",icon='😍')
            time.sleep(1)
            st.session_state["toast"] = 1
        # SVM model predicting personality
        if st.button("Predict",type="primary"):
            with open('gender_encoder.pkl', 'rb') as f:
                gender_encoder = pickle.load(f)
            with open('personality_encoder.pkl', 'rb') as f:
                personality_encoder = pickle.load(f)
            with open("personality_model.pkl", "rb") as f:
                model = pickle.load(f)
            with open("scaler.pkl", "rb") as f:
                scaler = pickle.load(f)
            feature_columns = ["Gender", "Age","openness","neuroticism","conscientiousness","agreeableness","extraversion"]
            column_data = [st.session_state["gender"], st.session_state["age"],st.session_state["openness"],st.session_state["neuroticism"],st.session_state["conscientiousness"],st.session_state["agreeableness"],st.session_state["extraversion"]]
            new_data = pd.DataFrame([column_data], columns=feature_columns)
            new_data['Gender'] = gender_encoder.transform(new_data['Gender'])
            new_data_scaled = scaler.transform(new_data)
            y_pred = model.predict(new_data_scaled)
            personality_pred = personality_encoder.inverse_transform(y_pred)[0]
            st.session_state["personality_pred"] = personality_pred
            st.rerun()
    # before completed questionnaire
    else:
        st.write('#### :gray[**Tell us about yourself**]')
        left,right = st.columns(2,gap="large")
        # Q1
        left.write('#### :rainbow[Gender]')
        gender = left.radio('Please select your gender.', ["Female","Male"])
        # Q2
        right.write('#### :rainbow[Age]')
        age = right.number_input("How old are you?", min_value=18, max_value=60,value=30)
        # Q3
        left.write('#### :rainbow[How open are you to trying new experiences and exploring different perspectives?]')
        openness = left.slider("On a scale from 1 to 8, where **1 means 'I prefer familiar routines'** and **8 means 'I actively seek out new and varied experiences'**", min_value=1, max_value=8,value=1)
        # Q4
        right.image("https://i.giphy.com/media/v1.Y2lkPTc5MGI3NjExc214dTNiOHEyaWE2bTF4bXdtZzJteDljcWxobXl4dWZ5OGN4bGdqeCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9cw/Vbu3WRl5LVfsJc4dSF/giphy.gif",use_column_width=True)
        right.write('#### :rainbow[How likely are you to feel upset by changes or challenges in your life?]')
        neuroticism = right.slider("On a scale from 1 to 8, where **1 means 'You rarely feel upset by changes or challenges.'** and **8 means 'You are quite sensitive to changes and challenges.'**", min_value=1, max_value=8,value=1)
        # Q5
        left.image("https://i.pinimg.com/736x/79/e0/16/79e0168825e6e5c8bb944615bf533984.jpg",use_column_width=True)
        left.write('#### :rainbow[How organized and goal-oriented are you in your daily life?]')
        conscientiousness = left.slider("On a scale from 1 to 8, where **1 means 'I tend to be spontaneous and go with the flow.'** and **8 means 'I am very organized and prefer to plan and stick to goals.'**", min_value=1, max_value=8,value=1)
        # Q6
        right.image("https://i.giphy.com/media/v1.Y2lkPTc5MGI3NjExdDJhbmFlMGdnMHNhMzVrdGZpejhsZWVlMGRzN3g2c3o0cnhscWhjYiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9cw/XbaxeGPpy0QIr4Cn7w/giphy.gif",use_column_width=True)
        right.write('#### :rainbow[How would you describe your tendency to empathize and cooperate with others?]')
        agreeableness = right.slider("On a scale from 1 to 8, where **1 means 'I tend to be more critical or assertive in my interactions.'** and **8 means 'I am highly empathetic and cooperative, often putting others' needs first.'**", min_value=1, max_value=8,value=1)
        # Q7
        left.image("https://i.pinimg.com/736x/9e/52/02/9e52025e26baf79d5eed38c96c296350.jpg",use_column_width=True)
        left.write('#### :rainbow[How do you feel about social interactions and being around other people?]')
        extraversion = left.slider("On a scale from 1 to 8, where **1 means 'I prefer solitude and quiet environments'** and **8 means 'I am energized by social gatherings and enjoy meeting new people.'**", min_value=1, max_value=8,value=1)
        right.image('https://i.pinimg.com/736x/a2/96/d6/a296d69276edafb7d2ad7b8d317a5314.jpg', use_column_width=True)
        # questionnaire submission
        if st.button('Submit',type="primary"):
            st.session_state["age"] = age
            st.session_state["gender"] = gender
            st.session_state["openness"] = openness
            st.session_state["neuroticism"] = neuroticism
            st.session_state["conscientiousness"] = conscientiousness
            st.session_state["agreeableness"] = agreeableness
            st.session_state["extraversion"] = extraversion
            st.rerun()

# personality result and genre recommendation
if  st.session_state["personality_pred"] is not None and st.session_state["genre_input"] is None:
    # personality result display
    left, right = st.columns(2,gap="large")
    left.markdown(
        """
        <h3 style="text-align: center; color: #d86cf5;">Personality result</h2>
        """,
        unsafe_allow_html=True
    )
    left.write(f"Your personality is <span style='color:red;'><b><i>{st.session_state["personality_pred"]}.", unsafe_allow_html=True)
    # personality picture display
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
    left.image(personality_image, width=300)

    # genre recommendation
    # genre recommendation module
    users_df = pd.read_csv('users.csv')
    top_genres = users_df[(users_df['age']==st.session_state["age"]) & (users_df['gender']==st.session_state["gender"])].groupby('selected_genre')['rating'].mean().sort_values(ascending=False).head(4).reset_index()
    top_genres = tuple(top_genres['selected_genre'])
    # Pick random genre choices
    book_genres_df = pd.read_csv('book_genres.csv')
    all_genres = tuple(book_genres_df['Genre'])
    remaining_genres = tuple(set(all_genres)-set(top_genres))
    random_fill = random.sample(remaining_genres, k=(5 - len(top_genres)))
    if st.session_state["genre_choices"] == ():
        st.session_state["genre_choices"] = top_genres + tuple(random_fill)
    # genre recommendation display
    right.image("https://i.giphy.com/media/v1.Y2lkPTc5MGI3NjExeWdzOWZ6bmh4ZTY3aDl1ZTdtb2xud25rMDMzaGRwb3l3a25iZnF6YiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9cw/29aPEcBjA5kbP1rbtx/giphy.gif",use_column_width=True)
    right.markdown(
        """
        <h3 style="text-align: center; color: #cd36f5;">Genre recommendation</h2>
        """,
        unsafe_allow_html=True
    )
    genre_input = right.selectbox("You may be interested in one of these genres. Pick one:", st.session_state["genre_choices"])
    if right.button('Submit'):
        st.session_state["genre_input"] = genre_input
        st.session_state["random_choice"] = None
        st.rerun()

# book recommendation
if st.session_state["genre_input"] is not None:
    # book recommender module
    book_df = pd.read_csv('book_genres.csv')
    url = book_df[book_df['Genre']==st.session_state["genre_input"]].iloc[0]['URL']        
    image_thumb = fetch_books_data(url)
    random_choice = random.choice(image_thumb)
    if st.session_state["random_choice"] == None:
        st.session_state["random_choice"] = random_choice
    random_element = st.session_state["random_choice"].find("img")
    book_url = f"https://nhanam.vn{st.session_state["random_choice"]['href']}"
    description = fetch_books_description(book_url)
    title = random_element.get("alt")
    # book recommendation display
    st.markdown(
        """
        <h3 style="text-align: center; color: #d86cf5;">Book recommendation</h2>
        """,
        unsafe_allow_html=True
    )
    left, right = st.columns(2, gap="large")
    left.write(f"##### <span style='color:red;'>{title}", unsafe_allow_html=True)
    left.image(random_element.get("data-src"),use_column_width=True)
    st.markdown(f"[Buy the book here]({book_url})")
    right.write(f"##### <span style='color:black;'>Description", unsafe_allow_html=True)
    right.write(description)
    # rating display
    if st.session_state["random_choice"].find("img").get("alt") in st.session_state["title"]:
        st.write("##### :purple[Have a second thought about this book?]")
    else:
        st.write("##### :rainbow[Do you like our recommendation?]")
    selected = st.feedback("stars")
    # 3 button set
    left, middle, right = st.columns(3)
    if left.button('Submit rating',use_container_width=True,type="primary"):
        new_row = [st.session_state["age"],st.session_state["gender"],st.session_state["personality_pred"],st.session_state["genre_input"],title,(selected+1),book_url,random_element.get("data-src")]
        with open('users.csv', mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(new_row)
        st.session_state["random_choice"] = None
        st.session_state["title"].append(title)
        st.rerun()
    if middle.button('Next book',use_container_width=True):
        st.session_state["random_choice"] = None
        st.rerun()
    if right.button('Change Genre',use_container_width=True):
        st.session_state["genre_input"] = None
        st.session_state["genre_choices"] = ()
        st.rerun()
    # top book suggestions
    # module and display
    users_df = pd.read_csv('users.csv')
    top_books = users_df[(users_df['age'] == st.session_state["age"]) & (users_df['gender'] == st.session_state["gender"])].groupby(['book','book_url','pic_url'])['rating'].mean().sort_values(ascending=False).head(3).reset_index()
    # st.write(top_books)
    if top_books.shape[0]==3:
        st.write("##### You may also like:")
        l, m, r = st.columns(3)
        l.image(top_books.iloc[0]['pic_url'],use_column_width=True)
        l.markdown(f"{top_books.iloc[0]['book']} ({top_books.iloc[0]['rating']} ⭐) - [BUY NOW]({top_books.iloc[0]['book_url']})")
        m.image(top_books.iloc[1]['pic_url'], use_column_width=True)
        m.markdown(f"{top_books.iloc[1]['book']} ({top_books.iloc[1]['rating']} ⭐) - [BUY NOW]({top_books.iloc[1]['book_url']})")
        r.image(top_books.iloc[2]['pic_url'], use_column_width=True)
        r.markdown(f"{top_books.iloc[2]['book']} ({top_books.iloc[2]['rating']} ⭐) - [BUY NOW]({top_books.iloc[2]['book_url']})")
    else:
        st.write("##### You are a great book adventurer!")


        
    
