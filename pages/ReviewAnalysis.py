import streamlit as st
import pandas as pd
import pandasql as psql
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from collections import defaultdict
import plotly.graph_objects as go
from plotly import tools
from plotly.subplots import make_subplots
import plotly.express as px
from plotly import __version__
import cufflinks as cf
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected = True)
cf.go_offline()
from wordcloud import WordCloud
import plotly.graph_objs as go
from plotly.offline import iplot
from collections import defaultdict
from wordcloud import STOPWORDS
from plotly import tools
hide_st_style="""
<style> #MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}
</style>"""
st.markdown(hide_st_style,unsafe_allow_html=True)
# Step 1: Allow user to upload CSV file
st.title("Product Review Analysis")
uploaded_file = st.file_uploader("Upload the CSV file obtained after the pre-processing done earlier with the columns Product_Title,Ratings,Review_Text", type="csv")
st.write("Note:It is advisible to use a dataset with less no of rows as time to classify may increase with more no of rows")
if uploaded_file is not None:
    dataframe = pd.read_csv(uploaded_file)
    dataframe['Brand'] = dataframe['Product_Title'].apply(lambda x: x.split()[0])
    dataframe['review_length'] = dataframe['Review_Text'].astype(str).apply(len)
    dataframe['word_count'] = dataframe['Review_Text'].apply(lambda x: len(str(x).split()))
    
    #Making an analysis of the based on the review length and word count of the user reviews for the products

    st.title("Review Length and Word Count Analysis based on reviews given by users")

    query = "SELECT Max(review_length),MAX(word_count) FROM dataframe"
    output = psql.sqldf(query, locals())
    st.write("Max review length and word count:", output)

    query = "SELECT AVG(review_length),AVG(word_count) FROM dataframe"
    output = psql.sqldf(query, locals())
    st.write("Average review length and word count:", output)

    query = "SELECT Min(review_length),MIN(word_count) FROM dataframe"
    output = psql.sqldf(query, locals())
    st.write("Min review length and word count:", output)

    query = "SELECT Brand,Max(review_length),MAX(word_count) FROM dataframe WHERE Brand<>'-------' GROUP BY Brand"
    output = psql.sqldf(query, locals())
    st.write("Max review length and word count by Brand:", output)

    query = "SELECT Brand,AVG(review_length),AVG(word_count) FROM dataframe WHERE Brand<>'-------' GROUP BY Brand"
    output = psql.sqldf(query, locals())
    st.write("Average review length and word count by Brand:", output)

    query = "SELECT Brand,MIN(review_length),MIN(word_count) FROM dataframe WHERE Brand<>'-------' GROUP BY Brand"
    output = psql.sqldf(query, locals())
    st.write("Min review length and word count by Brand:", output)
    fig1 = px.histogram(dataframe, x="review_length", nbins=100, title="Review Length Distribution")
    st.plotly_chart(fig1)
    fig2 = px.histogram(dataframe, x="word_count", nbins=100, title="Word Count Distribution")
    st.plotly_chart(fig2)
    st.title("Product Titles under each brand:")
    query = "SELECT Product_Title,Brand from dataframe group by Brand,Product_Title"
    grouped_products = psql.sqldf(query,locals())
    st.write(grouped_products)
 
    st.title("The Number of Product Titles for each brand")
    value_counts = dataframe['Brand'].value_counts()
    fig3 = px.bar(value_counts, x=value_counts.index, y=value_counts.values, title="Brand Frequency")
    st.plotly_chart(fig3)
    st.title("The Number of Distinct Brand names")
    query = "SELECT DISTINCT Brand FROM dataframe"
    uniquebrandnames = psql.sqldf(query, locals())
    st.write(uniquebrandnames)
    st.title("Reviews present for each product title of each brand")
    for i in uniquebrandnames['Brand']:
        query = f"SELECT Product_Title,Brand,Review_Text FROM dataframe WHERE Brand='{i}'"
        result = psql.sqldf(query, locals())
        st.write(f'{i}')
        st.write(result)
        st.write('------------')
    st.title("Word Cloud for Each Reviews of each distinct brand")
    for brand in uniquebrandnames['Brand']:
        brand_reviews = dataframe[dataframe['Brand'] == brand]['Review_Text']
    # Combine all review texts for the brand into one string
        st.write(f"Word Cloud for {brand}")
        text = " ".join(review for review in brand_reviews)
        wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=STOPWORDS).generate(text)
        st.image(wordcloud.to_array())
    st.write("Complete ReviewAnalysis by checking the box below, then proceed to SentimentAnalysis.")
        # task_1 Completion Checkbox
    ReviewAnalysis_complete = st.checkbox("Mark ReviewAnalysis as complete")
    if ReviewAnalysis_complete:
        st.session_state["ReviewAnalysis_complete"] = True
        st.success("ReviewAnalysis complete! You can proceed to SentimentAnalysis.")

    # Next button - only enabled if task_1 is complete
    if st.session_state.get("ReviewAnalysis_complete"):
        if st.button("Next Page"):
            st.session_state.current_page = "SentimentAnalysis"
    else:
        st.button("Next Page", disabled=True)


