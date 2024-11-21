from hmac import new
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.models import load_model
import pickle
import pandasql as psql
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re, string
import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px
from langdetect import detect
import io
st.title("Amazon Reviews Preprocessing Tool")
hide_st_style="""
<style> #MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}
</style>"""
st.markdown(hide_st_style,unsafe_allow_html=True)
# File upload
uploaded_file = st.file_uploader("Upload the reviews csv file you want to perform preprocessing on with the columns Product_Title,Title,Name,Ratings,Rating_Date,Review_Text", type="csv")
st.write("Note:It is advisible to use a dataset with less no of rows as time to classify may increase with more no of rows")
if uploaded_file is not None:
    newdf = pd.read_csv(uploaded_file)
    st.write("The reviews csv file you uploaded contains the following:")
    st.write(newdf)
    row_count = newdf.shape[0]
    st.write("Number of rows:", row_count)
    newdf = newdf.drop(['Unnamed: 0.1', 'Unnamed: 0'], axis=1).dropna()
    st.write("Columns present in your dataset:")
    st.write(f"{newdf.columns.tolist()}")
    newdf=newdf.apply(lambda x: x.astype(str).str.lower())
    st.write("Making all the text in the dataset lowercase, we get the following dataset:")
    st.write(newdf)
    newdf = newdf[newdf['Review_Text'] != '-------']
    st.write("The dataset after removing the rows with '-------'")
    st.write(newdf[['Product_Title','Title','Name','Ratings','Rating_Date','Review_Text']])
    row_count = newdf.shape[0]
    st.write("Number of rows:", row_count)
    #Training a LSTM model to classify the reviews into English or Hindi based on this dataset
    model = load_model(r'C:\Users\sujay\Desktop\Chatbot\models\Hinglish_Classification_LSTM_Model.h5')
    with open(r'C:\Users\sujay\Desktop\Chatbot\models\tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    def predict_label(review):
        sequence = tokenizer.texts_to_sequences([review])
        padded_sequence = pad_sequences(sequence, maxlen=200)
        prediction = model.predict(padded_sequence)
        sentiment = "en" if prediction[0][0] > 0.5 else "hi"
        return sentiment

    #newdf['EnglishorHindi'] = newdf['Review_Text'].apply(predict_label)
    newdf=pd.read_csv(r'C:\Users\sujay\Desktop\Chatbot\CSVFiles\langclass.csv')
    def classify_review_language(text, lang_code):
        if lang_code != 'en':  
            return 'foreign language'
        else:
            try:
                detected_lang = detect(text)  
                if detected_lang == 'en': 
                   return 'english'
                else:
                   return 'foreign language'
            except:
                 return 'unknown'
    #newdf['Language_Class'] = newdf.apply(lambda row: classify_review_language(row['Review_Text'], row['EnglishorHindi']), axis=1)
    st.write("After applying the language classification function to the dataset, we get the following results:")
    st.write(newdf)
    row_count = newdf.shape[0]
    st.write("Number of rows:", row_count)
    st.write("Distribution of foreign language to english reviews")
    value_counts = newdf['Language_Class'].value_counts()
    fig1 = px.bar(value_counts, x=value_counts.index, y=value_counts.values, title="Language in which review is madeCount based on each rating's Frequency")
    st.plotly_chart(fig1)
    newdf=newdf[newdf['Language_Class']=='english']
    newdf= newdf[['Product_Title', 'Ratings', 'Review_Text']]
    st.write("After removing any un-necessary columns and the rows from your dataset containing reviews in foreign languages, we get the following dataset:")
    st.write(newdf)
    row_count = newdf.shape[0]
    st.write("Number of rows:", row_count)
    st.write("-----------------")
    st.write("The frequency distribution of each column in the dataset is:")
    column_names=newdf.columns.tolist()
    for i in column_names:
        st.write(f"{i}")
        query = f"SELECT {i},count(*) FROM newdf group by {i}"
        result = psql.sqldf(query, locals())
        st.write(result)
        st.write('----------------------')
    value_counts = newdf['Ratings'].value_counts()
    fig1 = px.bar(value_counts, x=value_counts.index, y=value_counts.values, title="Ratings count based on each rating's Frequency")
    st.plotly_chart(fig1)
    newdf=newdf.drop_duplicates()
    stop_words = set(stopwords.words('english'))
    def remove_stopwords(text):
        if isinstance(text, str):
            return ' '.join([word for word in word_tokenize(text) if word.lower() not in stop_words])
        return text
    
    def remove_emojis(text):
        emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F" u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF" u"\U0001F700-\U0001F77F"
        u"\U0001F780-\U0001F7FF" u"\U0001F800-\U0001F8FF"
        u"\U0001F900-\U0001F9FF" u"\U0001FA00-\U0001FA6F"
        u"\U0001FA70-\U0001FAFF" u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251" "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', text)
    
    def remove_punctuation(text):
        return text.translate(str.maketrans('', '', string.punctuation))

    pattern = r'\b(media|video)\b'
    #newdf['Review_Text'] = newdf['Review_Text'].apply(lambda x: remove_stopwords(remove_emojis(remove_punctuation(x))))
    #newdf = newdf[~newdf['Review_Text'].str.contains(pattern, case=False, regex=True)].drop_duplicates()
    newdf=pd.read_csv(r"C:\Users\sujay\Desktop\Chatbot\CSVFiles\preprocessed.csv")
    st.write("The dataset after removing stopwords, emojis, punctuation, and rows containing 'media' or 'video':")
    st.write(newdf)
    row_count = newdf.shape[0]
    st.write("Number of rows:", row_count)
    st.title("Performing Exploratory Data Analysis on the finally obtaianed preprocessed dataset")
    st.write("The statistical inference made about the dataset is as follows:")
    st.write(newdf.describe())
    st.write('--------------')
    st.write("The number of duplicated rows is = ", newdf.duplicated().sum())
    st.write('--------------')
    st.write("The shape of the dataset is as follows:")
    st.write(newdf.shape)
    st.write('-------------')
    st.write("The data types of the dataset are as follows:")
    st.write(newdf.dtypes)
    st.write('-------------')
    st.write("The number of unique values in each column of the dataset are as follows:")
    st.write(newdf.nunique())
    st.write('---------------')
    st.write("The number of missing values in each column of the dataset are as follows:")
    st.write(newdf.isna().sum())
    st.write('---------------')
    # Save final preprocessed data
    st.write("Download the final preprocessed data:")
    csv_buffer = io.BytesIO()
    newdf.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)  # Go back to the start of the buffer
    st.download_button(label="Download the final preprocessed CSV File",data=csv_buffer,file_name="preprocessed_data.csv",mime="text/csv")
    st.write("After Downloading upload this file in the next page")
    st.write("Complete Preprocessing by checking the box below, then proceed to ReviewAnalysis.")
    Preprocessing_complete = st.checkbox("Mark Preprocessing as complete")
    if Preprocessing_complete:
        st.session_state["Preprocessing_complete"] = True
        st.success("Preprocessing complete! You can proceed to ReviewAnalysis.")

    if st.session_state.get("Preprocessing_complete"):
        if st.button("Next Page"):
            st.session_state.current_page = "ReviewAnalysis"
    else:
        st.button("Next Page", disabled=True)

    