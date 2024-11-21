import streamlit as st
import pandas as pd
from langchain_ollama import OllamaLLM
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
hide_st_style="""
<style> #MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}
</style>"""
# Initialize Llama 3 with Langchain
ollama = OllamaLLM(model="llama3")


st.title("Phone Review Insights with Llama 3")
st.write("Upload your dataset and select a product title to analyze reviews.")

# Step 1: Dataset Upload
uploaded_file = st.file_uploader("Upload any dataset containing phone reviews which has been classifed into with columns Sentiment - Positive/Negative, Reviews, Product_Title,Brand ", type="csv")

if uploaded_file is not None:
    # Step 2: Load and display dataset preview
    data = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:")
    st.write(data.head())

    # Step 3: Select product title
    product_titles = data['Product_Title'].unique()
    selected_product = st.selectbox("Select the Product Title:", product_titles)
    st.write(data[data['Product_Title'] == selected_product])
    # Step 4: Extract reviews for the selected product and categorize by sentiment
    if st.button("Analyze Reviews"):
        filtered_data = data[data['Product_Title'] == selected_product]
        
        # Split reviews based on sentiment
        positive_reviews = filtered_data[filtered_data['Sentiment'] == 'Positive']['Reviews'].tolist()
        negative_reviews = filtered_data[filtered_data['Sentiment'] == 'Negative']['Reviews'].tolist()
        
        # Join reviews into single strings
        combined_positive_reviews = " ".join(positive_reviews)
        combined_negative_reviews = " ".join(negative_reviews)
        
        # Step 5: Define a prompt template
        prompt_template = PromptTemplate(
            template=(
                "Analyze the following phone reviews for {product}. "
                "Based on these reviews, provide:\n1. Positive Features\n2. Negative Features\n3. Suggestions for Improvement\n\n"
                "Positive Reviews:\n{positive_reviews}\n\nNegative Reviews:\n{negative_reviews}"
            ),
            input_variables=["product", "positive_reviews", "negative_reviews"]
        )
        
        # Step 6: Create a chain that combines the prompt and the Ollama model
        chain = prompt_template|ollama
        
        # Step 7: Invoke the chain with the product title and reviews as input
        response = chain.invoke({
            "product": selected_product,
            "positive_reviews": combined_positive_reviews,
            "negative_reviews": combined_negative_reviews
        })
        
        # Step 8: Display the response
        st.subheader("Analysis Results")
        st.write(response)
    ChatBot_complete = st.checkbox("Mark  as complete")
    if ChatBot_complete:
            st.session_state["ChatBot_complete"] = True
            st.success("Entire Process is Completed!")
