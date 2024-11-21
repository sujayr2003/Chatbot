from hmac import new
import pandas as pd
from sklearn.svm import SVC
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, LSTM, Dense, Dropout,GlobalMaxPooling1D
from keras.layers import GRU,Bidirectional
from tensorflow.keras.optimizers import Adam
from sklearn import metrics
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score,confusion_matrix, roc_curve, auc
from streamlit_option_menu import option_menu
from transformers import pipeline
hide_st_style="""
<style> #MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}
</style>"""
st.markdown(hide_st_style,unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload csv file you just classified reviews into postive/negative sentiment based on ratings with columns Product_Title,Brand,Reviews,Ratings,Sentiment", type="csv")
st.write("Note:It is advisible to use a dataset with less no of rows as time to classify may increase with more no of rows")
if uploaded_file is not None:
    selected=option_menu(menu_title="Select the type of model you want to train/view results from",options=["Machine Learning","Deep Learning",'Hugging Face Transformer'],orientation="horizontal",)
    if selected == "Machine Learning":
        newdf = pd.read_csv(uploaded_file)
        st.write("### Raw Data Preview")
        st.write(newdf)
    # Dropdown menu
        option = st.selectbox(
        "Select a Model:",
        ["MultinomialNBClassifier", "LogisticRegressionClassifier", "DecisionTreeClassifier", "RandomForestClassifier", "KNeighborsClassifier","SupportVectorMachineClassifier"]
        )
    # Perform action based on the selected option
        if option == "MultinomialNBClassifier":
            data = newdf[['Reviews', 'Sentiment']]
            data['Sentiment'] = data['Sentiment'].apply(lambda x: 1 if x == 'Positive' else 0)

            splits = {
            "60-40": 0.4,
            "75-25": 0.25,
            "80-20": 0.2,
            "99-1": 0.01
            }

            results = []

            for split_name, test_size in splits.items():
                x_train, x_test, y_train, y_test = train_test_split(data['Reviews'], data['Sentiment'], test_size=test_size, random_state=42)

                # Vectorize the data
                cv = CountVectorizer()
                x_train_count = cv.fit_transform(x_train.values)
                x_test_count = cv.transform(x_test)

                # Train the model
                model = MultinomialNB()
                model.fit(x_train_count, y_train)

                # Predictions
                y_pred = model.predict(x_test_count)
                y_pred_prob = model.predict_proba(x_test_count)[:, 1]

                # Evaluate the model
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                accuracy = accuracy_score(y_test, y_pred)

                # Store results
                results.append([split_name, precision, recall, f1, accuracy])

                # For 80-20 split, plot confusion matrix and ROC curve
                if split_name == "80-20":
                    # Confusion matrix
                    cm = confusion_matrix(y_test, y_pred)
                    plt.figure(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
                    plt.xlabel('Predicted')
                    plt.ylabel('Actual')
                    plt.title('Confusion Matrix (80-20 Split)')
                    st.pyplot(plt)

                    # ROC curve
                    fpr, tpr, _ = roc_curve(y_test, y_pred)
                    roc_auc = auc(fpr, tpr)
                    plt.figure(figsize=(8, 6))
                    plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
                    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title('ROC Curve (80-20 Split)')
                    plt.legend(loc="lower right")
                    st.pyplot(plt)
                if split_name == "99-1":
                    st.write("Confusion Matrix and ROC Curve for 99-1 Split:")
                    cm = confusion_matrix(y_test, y_pred)
                    plt.figure(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
                    plt.xlabel('Predicted')
                    plt.ylabel('Actual')
                    plt.title('Confusion Matrix (99-1 Split)')
                    st.pyplot(plt)
                    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
                    roc_auc = auc(fpr, tpr)
                    plt.figure(figsize=(8, 6))
                    plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
                    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title('ROC Curve (99-1 Split)')
                    plt.legend(loc="lower right")
                    st.pyplot(plt)
                if split_name == "75-25":
                    st.write("Confusion Matrix and ROC Curve for 75-25 Split:")
                    cm = confusion_matrix(y_test, y_pred)
                    plt.figure(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
                    plt.xlabel('Predicted')
                    plt.ylabel('Actual')
                    plt.title('Confusion Matrix (75-25 Split)')
                    st.pyplot(plt)
                    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
                    roc_auc = auc(fpr, tpr)
                    plt.figure(figsize=(8, 6))
                    plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
                    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title('ROC Curve (75-25 Split)')
                    plt.legend(loc="lower right")
                    st.pyplot(plt)
                if split_name == "60-40":
                    st.write("Confusion Matrix and ROC Curve for 60-40 Split:")
                    cm = confusion_matrix(y_test, y_pred)
                    plt.figure(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
                    plt.xlabel('Predicted')
                    plt.ylabel('Actual')
                    plt.title('Confusion Matrix (60-40 Split)')
                    st.pyplot(plt)
                    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
                    roc_auc = auc(fpr, tpr)
                    plt.figure(figsize=(8, 6))
                    plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
                    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title('ROC Curve (60-40 Split)')
                    plt.legend(loc="lower right")
                    st.pyplot(plt)
            # Display evaluation results
            results_df = pd.DataFrame(results, columns=['Split', 'Precision', 'Recall', 'F1 Score', 'Accuracy'])
            st.write("Evaluation Metrics for Different Splits:")
            st.table(results_df)
            # Streamlit input for real-time prediction
            user_input = st.text_input("Enter a review:")
            if user_input:
               x_user_count = cv.transform([user_input])
               user_pred = model.predict(x_user_count)
               sentiment = "positive" if user_pred[0] == 1 else "negative"
               st.write(f"The predicted sentiment is: {sentiment}")
        if option == "LogisticRegressionClassifier":
            data = newdf[['Reviews', 'Sentiment']]
            data['Sentiment'] = data['Sentiment'].apply(lambda x: 1 if x == 'Positive' else 0)
            splits = {
            "60-40": 0.4,
            "75-25": 0.25,
            "80-20": 0.2,
            "99-1": 0.01
            }

            results = []

            for split_name, test_size in splits.items():
                x_train, x_test, y_train, y_test = train_test_split(data['Reviews'], data['Sentiment'], test_size=test_size, random_state=42)
                cv=CountVectorizer()
                x_train_count = cv.fit_transform(x_train.values)
                x_test_count = cv.transform(x_test)
                model=LogisticRegression()
                model.fit(x_train_count,y_train)
                y_pred = model.predict(x_test_count)
                y_pred_prob = model.predict_proba(x_test_count)[:, 1]
                # Evaluate the model
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                accuracy = accuracy_score(y_test, y_pred)

                # Store results
                results.append([split_name, precision, recall, f1, accuracy])

                # For 80-20 split, plot confusion matrix and ROC curve
                if split_name == "80-20":
                    # Confusion matrix
                    cm = confusion_matrix(y_test, y_pred)
                    plt.figure(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
                    plt.xlabel('Predicted')
                    plt.ylabel('Actual')
                    plt.title('Confusion Matrix (80-20 Split)')
                    st.pyplot(plt)

                    # ROC curve
                    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
                    roc_auc = auc(fpr, tpr)
                    plt.figure(figsize=(8, 6))
                    plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
                    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title('ROC Curve (80-20 Split)')
                    plt.legend(loc="lower right")
                    st.pyplot(plt)
                if split_name == "99-1":
                    st.write("Confusion Matrix and ROC Curve for 99-1 Split:")
                    cm = confusion_matrix(y_test, y_pred)
                    plt.figure(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
                    plt.xlabel('Predicted')
                    plt.ylabel('Actual')
                    plt.title('Confusion Matrix (99-1 Split)')
                    st.pyplot(plt)
                    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
                    roc_auc = auc(fpr, tpr)
                    plt.figure(figsize=(8, 6))
                    plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
                    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title('ROC Curve (99-1 Split)')
                    plt.legend(loc="lower right")
                    st.pyplot(plt)
                if split_name == "75-25":
                    st.write("Confusion Matrix and ROC Curve for 75-25 Split:")
                    cm = confusion_matrix(y_test, y_pred)
                    plt.figure(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
                    plt.xlabel('Predicted')
                    plt.ylabel('Actual')
                    plt.title('Confusion Matrix (75-25 Split)')
                    st.pyplot(plt)
                    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
                    roc_auc = auc(fpr, tpr)
                    plt.figure(figsize=(8, 6))
                    plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
                    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title('ROC Curve (75-25 Split)')
                    plt.legend(loc="lower right")
                    st.pyplot(plt)
                if split_name == "60-40":
                    st.write("Confusion Matrix and ROC Curve for 60-40 Split:")
                    cm = confusion_matrix(y_test, y_pred)
                    plt.figure(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
                    plt.xlabel('Predicted')
                    plt.ylabel('Actual')
                    plt.title('Confusion Matrix (60-40 Split)')
                    st.pyplot(plt)
                    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
                    roc_auc = auc(fpr, tpr)
                    plt.figure(figsize=(8, 6))
                    plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
                    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title('ROC Curve (60-40 Split)')
                    plt.legend(loc="lower right")
                    st.pyplot(plt)
            # Display evaluation results
            results_df = pd.DataFrame(results, columns=['Split', 'Precision', 'Recall', 'F1 Score', 'Accuracy'])
            st.write("Evaluation Metrics for Different Splits:")
            st.table(results_df)
            # Streamlit input for real-time prediction
            user_input = st.text_input("Enter a review:")
            if user_input:
               x_user_count = cv.transform([user_input])
               user_pred = model.predict(x_user_count)
               sentiment = "positive" if user_pred[0] == 1 else "negative"
               st.write(f"The predicted sentiment is: {sentiment}")
        if option == "DecisionTreeClassifier":
            data = newdf[['Reviews', 'Sentiment']]
            data['Sentiment'] = data['Sentiment'].apply(lambda x: 1 if x == 'Positive' else 0)
            splits = {
            "60-40": 0.4,
            "75-25": 0.25,
            "80-20": 0.2,
            "99-1": 0.01
            }
            results = []
            for split_name, test_size in splits.items():
                x_train, x_test, y_train, y_test = train_test_split(data['Reviews'], data['Sentiment'], test_size=test_size, random_state=42)
                cv=CountVectorizer()
                x_train_count = cv.fit_transform(x_train.values)
                x_test_count = cv.transform(x_test)
                model=DecisionTreeClassifier()
                model.fit(x_train_count,y_train)
                y_pred = model.predict(x_test_count)
                y_pred_prob = model.predict_proba(x_test_count)[:, 1]
                # Evaluate the model
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                accuracy = accuracy_score(y_test, y_pred)
                # Store results
                results.append([split_name, precision, recall, f1, accuracy])

                # For 80-20 split, plot confusion matrix and ROC curve
                if split_name == "80-20":
                    # Confusion matrix
                    cm = confusion_matrix(y_test, y_pred)
                    plt.figure(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
                    plt.xlabel('Predicted')
                    plt.ylabel('Actual')
                    plt.title('Confusion Matrix (80-20 Split)')
                    st.pyplot(plt)

                    # ROC curve
                    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
                    roc_auc = auc(fpr, tpr)
                    plt.figure(figsize=(8, 6))
                    plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
                    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title('ROC Curve (80-20 Split)')
                    plt.legend(loc="lower right")
                    st.pyplot(plt)
                if split_name == "99-1":
                    st.write("Confusion Matrix and ROC Curve for 99-1 Split:")
                    cm = confusion_matrix(y_test, y_pred)
                    plt.figure(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
                    plt.xlabel('Predicted')
                    plt.ylabel('Actual')
                    plt.title('Confusion Matrix (99-1 Split)')
                    st.pyplot(plt)
                    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
                    roc_auc = auc(fpr, tpr)
                    plt.figure(figsize=(8, 6))
                    plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
                    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title('ROC Curve (99-1 Split)')
                    plt.legend(loc="lower right")
                    st.pyplot(plt)
                if split_name == "75-25":
                    st.write("Confusion Matrix and ROC Curve for 75-25 Split:")
                    cm = confusion_matrix(y_test, y_pred)
                    plt.figure(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
                    plt.xlabel('Predicted')
                    plt.ylabel('Actual')
                    plt.title('Confusion Matrix (75-25 Split)')
                    st.pyplot(plt)
                    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
                    roc_auc = auc(fpr, tpr)
                    plt.figure(figsize=(8, 6))
                    plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
                    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title('ROC Curve (75-25 Split)')
                    plt.legend(loc="lower right")
                    st.pyplot(plt)
                if split_name == "60-40":
                    st.write("Confusion Matrix and ROC Curve for 60-40 Split:")
                    cm = confusion_matrix(y_test, y_pred)
                    plt.figure(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
                    plt.xlabel('Predicted')
                    plt.ylabel('Actual')
                    plt.title('Confusion Matrix (60-40 Split)')
                    st.pyplot(plt)
                    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
                    roc_auc = auc(fpr, tpr)
                    plt.figure(figsize=(8, 6))
                    plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
                    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title('ROC Curve (60-40 Split)')
                    plt.legend(loc="lower right")
                    st.pyplot(plt)
            # Display evaluation results
            results_df = pd.DataFrame(results, columns=['Split', 'Precision', 'Recall', 'F1 Score', 'Accuracy'])
            st.write("Evaluation Metrics for Different Splits:")
            st.table(results_df)
            # Streamlit input for real-time prediction
            user_input = st.text_input("Enter a review:")
            if user_input:
               x_user_count = cv.transform([user_input])
               user_pred = model.predict(x_user_count)
               sentiment = "positive" if user_pred[0] == 1 else "negative"
               st.write(f"The predicted sentiment is: {sentiment}")

        if option == "RandomForestClassifier":
            data = newdf[['Reviews', 'Sentiment']]
            data['Sentiment'] = data['Sentiment'].apply(lambda x: 1 if x == 'Positive' else 0)
            splits = {
            "60-40": 0.4,
            "75-25": 0.25,
            "80-20": 0.2,
            "99-1": 0.01
            }
            results = []

            for split_name, test_size in splits.items():
                x_train, x_test, y_train, y_test = train_test_split(data['Reviews'], data['Sentiment'], test_size=test_size, random_state=42)
                cv=CountVectorizer()
                x_train_count = cv.fit_transform(x_train.values)
                x_test_count = cv.transform(x_test)
                model=RandomForestClassifier()
                model.fit(x_train_count,y_train)
                y_pred = model.predict(x_test_count)
                y_pred_prob = model.predict_proba(x_test_count)[:, 1]
                # Evaluate the model
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                accuracy = accuracy_score(y_test, y_pred)
                # Store results
                results.append([split_name, precision, recall, f1, accuracy])
                if split_name == "80-20":
                    # Confusion matrix
                    cm = confusion_matrix(y_test, y_pred)
                    plt.figure(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
                    plt.xlabel('Predicted')
                    plt.ylabel('Actual')
                    plt.title('Confusion Matrix (80-20 Split)')
                    st.pyplot(plt)

                    # ROC curve
                    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
                    roc_auc = auc(fpr, tpr)
                    plt.figure(figsize=(8, 6))
                    plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
                    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title('ROC Curve (80-20 Split)')
                    plt.legend(loc="lower right")
                    st.pyplot(plt)
                if split_name == "99-1":
                    st.write("Confusion Matrix and ROC Curve for 99-1 Split:")
                    cm = confusion_matrix(y_test, y_pred)
                    plt.figure(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
                    plt.xlabel('Predicted')
                    plt.ylabel('Actual')
                    plt.title('Confusion Matrix (99-1 Split)')
                    st.pyplot(plt)
                    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
                    roc_auc = auc(fpr, tpr)
                    plt.figure(figsize=(8, 6))
                    plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
                    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title('ROC Curve (99-1 Split)')
                    plt.legend(loc="lower right")
                    st.pyplot(plt)
                if split_name == "75-25":
                    st.write("Confusion Matrix and ROC Curve for 75-25 Split:")
                    cm = confusion_matrix(y_test, y_pred)
                    plt.figure(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
                    plt.xlabel('Predicted')
                    plt.ylabel('Actual')
                    plt.title('Confusion Matrix (75-25 Split)')
                    st.pyplot(plt)
                    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
                    roc_auc = auc(fpr, tpr)
                    plt.figure(figsize=(8, 6))
                    plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
                    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title('ROC Curve (75-25 Split)')
                    plt.legend(loc="lower right")
                    st.pyplot(plt)
                if split_name == "60-40":
                    st.write("Confusion Matrix and ROC Curve for 60-40 Split:")
                    cm = confusion_matrix(y_test, y_pred)
                    plt.figure(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
                    plt.xlabel('Predicted')
                    plt.ylabel('Actual')
                    plt.title('Confusion Matrix (60-40 Split)')
                    st.pyplot(plt)
                    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
                    roc_auc = auc(fpr, tpr)
                    plt.figure(figsize=(8, 6))
                    plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
                    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title('ROC Curve (60-40 Split)')
                    plt.legend(loc="lower right")
                    st.pyplot(plt)
            # Display evaluation results
            results_df = pd.DataFrame(results, columns=['Split', 'Precision', 'Recall', 'F1 Score', 'Accuracy'])
            st.write("Evaluation Metrics for Different Splits:")
            st.table(results_df)
            # Streamlit input for real-time prediction
            user_input = st.text_input("Enter a review:")
            if user_input:
               x_user_count = cv.transform([user_input])
               user_pred = model.predict(x_user_count)
               sentiment = "positive" if user_pred[0] == 1 else "negative"
               st.write(f"The predicted sentiment is: {sentiment}")
        if option == "KNeighborsClassifier":
            data = newdf[['Reviews', 'Sentiment']]
            data['Sentiment'] = data['Sentiment'].apply(lambda x: 1 if x == 'Positive' else 0)
            splits = {
            "60-40": 0.4,
            "75-25": 0.25,
            "80-20": 0.2,
            "99-1": 0.01
            }
            results = []

            for split_name, test_size in splits.items():
                x_train, x_test, y_train, y_test = train_test_split(data['Reviews'], data['Sentiment'], test_size=test_size, random_state=42)
                cv=CountVectorizer()
                x_train_count = cv.fit_transform(x_train.values)
                x_test_count = cv.transform(x_test)
                model=KNeighborsClassifier()
                model.fit(x_train_count,y_train)
                y_pred = model.predict(x_test_count)
                y_pred_prob = model.predict_proba(x_test_count)[:, 1]
                # Evaluate the model
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                accuracy = accuracy_score(y_test, y_pred)
                # Store results
                results.append([split_name, precision, recall, f1, accuracy])
                if split_name == "80-20":
                    # Confusion matrix
                    cm = confusion_matrix(y_test, y_pred)
                    plt.figure(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
                    plt.xlabel('Predicted')
                    plt.ylabel('Actual')
                    plt.title('Confusion Matrix (80-20 Split)')
                    st.pyplot(plt)

                    # ROC curve
                    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
                    roc_auc = auc(fpr, tpr)
                    plt.figure(figsize=(8, 6))
                    plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
                    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title('ROC Curve (80-20 Split)')
                    plt.legend(loc="lower right")
                    st.pyplot(plt)
                if split_name == "99-1":
                    st.write("Confusion Matrix and ROC Curve for 99-1 Split:")
                    cm = confusion_matrix(y_test, y_pred)
                    plt.figure(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
                    plt.xlabel('Predicted')
                    plt.ylabel('Actual')
                    plt.title('Confusion Matrix (99-1 Split)')
                    st.pyplot(plt)
                    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
                    roc_auc = auc(fpr, tpr)
                    plt.figure(figsize=(8, 6))
                    plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
                    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title('ROC Curve (99-1 Split)')
                    plt.legend(loc="lower right")
                    st.pyplot(plt)
                if split_name == "75-25":
                    st.write("Confusion Matrix and ROC Curve for 75-25 Split:")
                    cm = confusion_matrix(y_test, y_pred)
                    plt.figure(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
                    plt.xlabel('Predicted')
                    plt.ylabel('Actual')
                    plt.title('Confusion Matrix (75-25 Split)')
                    st.pyplot(plt)
                    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
                    roc_auc = auc(fpr, tpr)
                    plt.figure(figsize=(8, 6))
                    plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
                    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title('ROC Curve (75-25 Split)')
                    plt.legend(loc="lower right")
                    st.pyplot(plt)
                if split_name == "60-40":
                    st.write("Confusion Matrix and ROC Curve for 60-40 Split:")
                    cm = confusion_matrix(y_test, y_pred)
                    plt.figure(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
                    plt.xlabel('Predicted')
                    plt.ylabel('Actual')
                    plt.title('Confusion Matrix (60-40 Split)')
                    st.pyplot(plt)
                    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
                    roc_auc = auc(fpr, tpr)
                    plt.figure(figsize=(8, 6))
                    plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
                    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title('ROC Curve (60-40 Split)')
                    plt.legend(loc="lower right")
                    st.pyplot(plt)
            # Display evaluation results
            results_df = pd.DataFrame(results, columns=['Split', 'Precision', 'Recall', 'F1 Score', 'Accuracy'])
            st.write("Evaluation Metrics for Different Splits:")
            st.table(results_df)
            # Streamlit input for real-time prediction
            user_input = st.text_input("Enter a review:")
            if user_input:
               x_user_count = cv.transform([user_input])
               user_pred = model.predict(x_user_count)
               sentiment = "positive" if user_pred[0] == 1 else "negative"
               st.write(f"The predicted sentiment is: {sentiment}")

        if option == "SupportVectorMachineClassifier":
            data = newdf[['Reviews', 'Sentiment']]
            data['Sentiment'] = data['Sentiment'].apply(lambda x: 1 if x == 'Positive' else 0)
            splits = {
            "60-40": 0.4,
            "75-25": 0.25,
            "80-20": 0.2,
            "99-1": 0.01
            }
            results = []

            for split_name, test_size in splits.items():

                x_train, x_test, y_train, y_test = train_test_split(data['Reviews'], data['Sentiment'], test_size=test_size, random_state=42)
                cv=CountVectorizer()
                x_train_count = cv.fit_transform(x_train.values)
                x_test_count = cv.transform(x_test)
                model = SVC()
                model.fit(x_train_count, y_train)
                y_pred = model.predict(x_test_count)
                #y_pred_prob = model.predict_proba(x_test_count)[:, 1]
                # Evaluate the model
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                accuracy = accuracy_score(y_test, y_pred)
                # Store results
                results.append([split_name, precision, recall, f1, accuracy])
                if split_name == "80-20":
                    # Confusion matrix
                    cm = confusion_matrix(y_test, y_pred)
                    plt.figure(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
                    plt.xlabel('Predicted')
                    plt.ylabel('Actual')
                    plt.title('Confusion Matrix (80-20 Split)')
                    st.pyplot(plt)

                    # ROC curve
                    fpr, tpr, _ = roc_curve(y_test, y_pred)
                    roc_auc = auc(fpr, tpr)
                    plt.figure(figsize=(8, 6))
                    plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
                    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title('ROC Curve (80-20 Split)')
                    plt.legend(loc="lower right")
                    st.pyplot(plt)
                if split_name == "99-1":
                    st.write("Confusion Matrix and ROC Curve for 99-1 Split:")
                    cm = confusion_matrix(y_test, y_pred)
                    plt.figure(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
                    plt.xlabel('Predicted')
                    plt.ylabel('Actual')
                    plt.title('Confusion Matrix (99-1 Split)')
                    st.pyplot(plt)
                    fpr, tpr, _ = roc_curve(y_test, y_pred)
                    roc_auc = auc(fpr, tpr)
                    plt.figure(figsize=(8, 6))
                    plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
                    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title('ROC Curve (99-1 Split)')
                    plt.legend(loc="lower right")
                    st.pyplot(plt)
                if split_name == "75-25":
                    st.write("Confusion Matrix and ROC Curve for 75-25 Split:")
                    cm = confusion_matrix(y_test, y_pred)
                    plt.figure(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
                    plt.xlabel('Predicted')
                    plt.ylabel('Actual')
                    plt.title('Confusion Matrix (75-25 Split)')
                    st.pyplot(plt)
                    fpr, tpr, _ = roc_curve(y_test, y_pred)
                    roc_auc = auc(fpr, tpr)
                    plt.figure(figsize=(8, 6))
                    plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
                    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title('ROC Curve (75-25 Split)')
                    plt.legend(loc="lower right")
                    st.pyplot(plt)
                if split_name == "60-40":
                    st.write("Confusion Matrix and ROC Curve for 60-40 Split:")
                    cm = confusion_matrix(y_test, y_pred)
                    plt.figure(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
                    plt.xlabel('Predicted')
                    plt.ylabel('Actual')
                    plt.title('Confusion Matrix (60-40 Split)')
                    st.pyplot(plt)
                    fpr, tpr, _ = roc_curve(y_test, y_pred)
                    roc_auc = auc(fpr, tpr)
                    plt.figure(figsize=(8, 6))
                    plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
                    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title('ROC Curve (60-40 Split)')
                    plt.legend(loc="lower right")
                    st.pyplot(plt)
            # Display evaluation results
            results_df = pd.DataFrame(results, columns=['Split', 'Precision', 'Recall', 'F1 Score', 'Accuracy'])
            st.write("Evaluation Metrics for Different Splits:")
            st.table(results_df)
            # Streamlit input for real-time prediction
            user_input = st.text_input("Enter a review:")
            if user_input:
               x_user_count = cv.transform([user_input])
               user_pred = model.predict(x_user_count)
               sentiment = "positive" if user_pred[0] == 1 else "negative"
               st.write(f"The predicted sentiment is: {sentiment}")
    if selected == "Deep Learning":
        newdf = pd.read_csv(uploaded_file)
        st.write("### Raw Data Preview")
        st.write(newdf)
        option = st.selectbox(
        "Select a Model:",
        ["LSTM", "RecurrentandConvolutionalNeuralNetwork", "ConvolutionalNeuralNetwork", "GateRecurrentUnit", "BidirectionalLSTM"]
        )
        if option == "LSTM":
            data = newdf[['Reviews', 'Sentiment']]
            data.replace({"Sentiment": {"Positive": 1, "Negative": 0}}, inplace=True)

            # Define different train-test split ratios
            split_ratios = {
                "80-20": 0.2,
                "99-1": 0.01,
                "75-25": 0.25,
                "60-40": 0.4
            }

            # Initialize table to store results
            results = []

            for split_name, test_size in split_ratios.items():
                train_data, test_data = train_test_split(data, test_size=test_size, random_state=42)
                tokenizer = Tokenizer(num_words=10000)
                tokenizer.fit_on_texts(train_data["Reviews"])
                X_train = pad_sequences(tokenizer.texts_to_sequences(train_data["Reviews"]), maxlen=200)
                X_test = pad_sequences(tokenizer.texts_to_sequences(test_data["Reviews"]), maxlen=200)
                Y_train = train_data["Sentiment"].values
                Y_test = test_data["Sentiment"].values
                model = Sequential([
                    Embedding(input_dim=10000, output_dim=128, input_length=200),
                    LSTM(128, dropout=0.2, recurrent_dropout=0.2),
                    Dense(1, activation="sigmoid")
                ])
                model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])
                model.fit(X_train, Y_train, epochs=5, batch_size=32, validation_split=0.2, verbose=1)
                y_pred_prob = model.predict(X_test).flatten()
                y_pred_classes = (y_pred_prob > 0.5).astype(int)
                precision = precision_score(Y_test, y_pred_classes)
                recall = recall_score(Y_test, y_pred_classes)
                f1 = f1_score(Y_test, y_pred_classes)
                accuracy = accuracy_score(Y_test, y_pred_classes)
                results.append({
                    "Split": split_name,
                    "Precision": precision,
                    "Recall": recall,
                    "F1 Score": f1,
                    "Accuracy": accuracy
                })
                if split_name == "80-20":
                    # Confusion Matrix
                    st.write("Confusion Matrix and ROC Curve for 80-20 Split:")
                    cm = confusion_matrix(Y_test, y_pred_classes)
                    plt.figure(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
                    plt.xlabel('Predicted')
                    plt.ylabel('Actual')
                    plt.title('Confusion Matrix (80-20 Split)')
                    st.pyplot(plt)
                    fpr, tpr, _ = roc_curve(Y_test, y_pred_prob)
                    roc_auc = auc(fpr, tpr)
                    plt.figure(figsize=(8, 6))
                    plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
                    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title('ROC Curve (80-20 Split)')
                    plt.legend(loc="lower right")
                    st.pyplot(plt)
                if split_name == "99-1":
                    st.write("Confusion Matrix and ROC Curve for 99-1 Split:")
                    cm = confusion_matrix(Y_test, y_pred_classes)
                    plt.figure(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
                    plt.xlabel('Predicted')
                    plt.ylabel('Actual')
                    plt.title('Confusion Matrix (99-1 Split)')
                    st.pyplot(plt)
                    fpr, tpr, _ = roc_curve(Y_test, y_pred_prob)
                    roc_auc = auc(fpr, tpr)
                    plt.figure(figsize=(8, 6))
                    plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
                    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title('ROC Curve (99-1 Split)')
                    plt.legend(loc="lower right")
                    st.pyplot(plt)
                if split_name == "75-25":
                    st.write("Confusion Matrix and ROC Curve for 75-25 Split:")
                    cm = confusion_matrix(Y_test, y_pred_classes)
                    plt.figure(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
                    plt.xlabel('Predicted')
                    plt.ylabel('Actual')
                    plt.title('Confusion Matrix (75-25 Split)')
                    st.pyplot(plt)
                    fpr, tpr, _ = roc_curve(Y_test, y_pred_prob)
                    roc_auc = auc(fpr, tpr)
                    plt.figure(figsize=(8, 6))
                    plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
                    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title('ROC Curve (75-25 Split)')
                    plt.legend(loc="lower right")
                    st.pyplot(plt)
                if split_name == "60-40":
                    st.write("Confusion Matrix and ROC Curve for 60-40 Split:")
                    cm = confusion_matrix(Y_test, y_pred_classes)
                    plt.figure(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
                    plt.xlabel('Predicted')
                    plt.ylabel('Actual')
                    plt.title('Confusion Matrix (60-40 Split)')
                    st.pyplot(plt)
                    fpr, tpr, _ = roc_curve(Y_test, y_pred_prob)
                    roc_auc = auc(fpr, tpr)
                    plt.figure(figsize=(8, 6))
                    plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
                    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title('ROC Curve (60-40 Split)')
                    plt.legend(loc="lower right")
                    st.pyplot(plt)
            results_df = pd.DataFrame(results)
            st.write("Evaluation Metrics for Different Splits:")
            st.table(results_df)
            if 'lstm_model' not in st.session_state:
                st.session_state['lstm_model'] = model
                st.session_state['tokenizer'] = tokenizer
            def predict_sentiment(review):
                sequence = st.session_state['tokenizer'].texts_to_sequences([review])
                padded_sequence = pad_sequences(sequence, maxlen=200)
                prediction = st.session_state['lstm_model'].predict(padded_sequence)
                sentiment = "positive" if prediction[0][0] > 0.5 else "negative"
                return sentiment
            user_input = st.text_input("Enter a review:")
            if user_input:
                sentiment = predict_sentiment(user_input)
                st.write(f"The predicted sentiment is: {sentiment}")
        if option == "RecurrentandConvolutionalNeuralNetwork":
            data = newdf[['Reviews', 'Sentiment']]
            data.replace({"Sentiment": {"Positive": 1, "Negative": 0}}, inplace=True)

            # Define different train-test split ratios
            split_ratios = {
                "80-20": 0.2,
                "99-1": 0.01,
                "75-25": 0.25,
                "60-40": 0.4
            }

            results = []

            for split_name, test_size in split_ratios.items():
                train_data, test_data = train_test_split(data, test_size=test_size, random_state=42)
                tokenizer = Tokenizer(num_words=10000)
                tokenizer.fit_on_texts(train_data["Reviews"])
                X_train = pad_sequences(tokenizer.texts_to_sequences(train_data["Reviews"]), maxlen=200)
                X_test = pad_sequences(tokenizer.texts_to_sequences(test_data["Reviews"]), maxlen=200)
                Y_train = train_data["Sentiment"]
                Y_test = test_data["Sentiment"]
                # Define Hybrid CNN-LSTM Model
                model = Sequential([
                Embedding(input_dim=10000, output_dim=128, input_length=200),
                Conv1D(filters=64, kernel_size=3, activation='relu'),
                MaxPooling1D(pool_size=2),
                LSTM(128, dropout=0.2, recurrent_dropout=0.2),
                Dense(1, activation='sigmoid')
                ])
                model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                model.fit(X_train, Y_train, epochs=5, batch_size=32, validation_split=0.2)
                y_pred_prob = model.predict(X_test).flatten()
                y_pred_classes = (y_pred_prob > 0.5).astype(int)
                precision = precision_score(Y_test, y_pred_classes)
                recall = recall_score(Y_test, y_pred_classes)
                f1 = f1_score(Y_test, y_pred_classes)
                accuracy = accuracy_score(Y_test, y_pred_classes)
                results.append({
                    "Split": split_name,
                    "Precision": precision,
                    "Recall": recall,
                    "F1 Score": f1,
                    "Accuracy": accuracy
                })
                if split_name == "80-20":
                    # Confusion Matrix
                    st.write("Confusion Matrix and ROC Curve for 80-20 Split:")
                    cm = confusion_matrix(Y_test, y_pred_classes)
                    plt.figure(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
                    plt.xlabel('Predicted')
                    plt.ylabel('Actual')
                    plt.title('Confusion Matrix (80-20 Split)')
                    st.pyplot(plt)
                    fpr, tpr, _ = roc_curve(Y_test, y_pred_prob)
                    roc_auc = auc(fpr, tpr)
                    plt.figure(figsize=(8, 6))
                    plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
                    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title('ROC Curve (80-20 Split)')
                    plt.legend(loc="lower right")
                    st.pyplot(plt)
                if split_name == "99-1":
                    st.write("Confusion Matrix and ROC Curve for 99-1 Split:")
                    cm = confusion_matrix(Y_test, y_pred_classes)
                    plt.figure(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
                    plt.xlabel('Predicted')
                    plt.ylabel('Actual')
                    plt.title('Confusion Matrix (99-1 Split)')
                    st.pyplot(plt)
                    fpr, tpr, _ = roc_curve(Y_test, y_pred_prob)
                    roc_auc = auc(fpr, tpr)
                    plt.figure(figsize=(8, 6))
                    plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
                    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title('ROC Curve (99-1 Split)')
                    plt.legend(loc="lower right")
                    st.pyplot(plt)
                if split_name == "75-25":
                    st.write("Confusion Matrix and ROC Curve for 75-25 Split:")
                    cm = confusion_matrix(Y_test, y_pred_classes)
                    plt.figure(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
                    plt.xlabel('Predicted')
                    plt.ylabel('Actual')
                    plt.title('Confusion Matrix (75-25 Split)')
                    st.pyplot(plt)
                    fpr, tpr, _ = roc_curve(Y_test, y_pred_prob)
                    roc_auc = auc(fpr, tpr)
                    plt.figure(figsize=(8, 6))
                    plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
                    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title('ROC Curve (75-25 Split)')
                    plt.legend(loc="lower right")
                    st.pyplot(plt)
                if split_name == "60-40":
                    st.write("Confusion Matrix and ROC Curve for 60-40 Split:")
                    cm = confusion_matrix(Y_test, y_pred_classes)
                    plt.figure(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
                    plt.xlabel('Predicted')
                    plt.ylabel('Actual')
                    plt.title('Confusion Matrix (60-40 Split)')
                    st.pyplot(plt)
                    fpr, tpr, _ = roc_curve(Y_test, y_pred_prob)
                    roc_auc = auc(fpr, tpr)
                    plt.figure(figsize=(8, 6))
                    plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
                    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title('ROC Curve (60-40 Split)')
                    plt.legend(loc="lower right")
                    st.pyplot(plt)
            results_df = pd.DataFrame(results)
            st.write("Evaluation Metrics for Different Splits:")
            st.table(results_df)
            if 'rnnandcnn_model' not in st.session_state:
                st.session_state['rnnandcnn_model'] = model
                st.session_state['tokenizer'] = tokenizer
            def predict_sentiment(review):
                sequence = st.session_state['tokenizer'].texts_to_sequences([review])
                padded_sequence = pad_sequences(sequence, maxlen=200)
                prediction = st.session_state['rnnandcnn_model'].predict(padded_sequence)
                sentiment = "positive" if prediction[0][0] > 0.5 else "negative"
                return sentiment
            user_input = st.text_input("Enter a review:")
            if user_input:
                sentiment = predict_sentiment(user_input)
                st.write(f"The predicted sentiment is: {sentiment}")
        if option == "ConvolutionalNeuralNetwork":
            data = newdf[['Reviews', 'Sentiment']]
            data.replace({"Sentiment": {"Positive": 1, "Negative": 0}}, inplace=True)

            # Define different train-test split ratios
            split_ratios = {
                "80-20": 0.2,
                "99-1": 0.01,
                "75-25": 0.25,
                "60-40": 0.4
            }

            results = []

            for split_name, test_size in split_ratios.items():
                train_data, test_data = train_test_split(data, test_size=test_size, random_state=42)
                tokenizer = Tokenizer(num_words=10000)
                tokenizer.fit_on_texts(train_data["Reviews"])
                X_train = pad_sequences(tokenizer.texts_to_sequences(train_data["Reviews"]), maxlen=200)
                X_test = pad_sequences(tokenizer.texts_to_sequences(test_data["Reviews"]), maxlen=200)
                Y_train = train_data["Sentiment"]
                Y_test = test_data["Sentiment"]
                max_words = 10000
                max_len = 200
                model = Sequential()
                model.add(Embedding(max_words, 128, input_length=max_len))
                model.add(Conv1D(128, 5, activation='relu'))
                model.add(GlobalMaxPooling1D())
                model.add(Dense(10, activation='relu'))
                model.add(Dense(1, activation='sigmoid'))
                model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                model.fit(X_train, Y_train, epochs=5, batch_size=32, validation_split=0.2)
                y_pred_prob = model.predict(X_test).flatten()
                y_pred_classes = (y_pred_prob > 0.5).astype(int)
                precision = precision_score(Y_test, y_pred_classes)
                recall = recall_score(Y_test, y_pred_classes)
                f1 = f1_score(Y_test, y_pred_classes)
                accuracy = accuracy_score(Y_test, y_pred_classes)
                results.append({
                    "Split": split_name,
                    "Precision": precision,
                    "Recall": recall,
                    "F1 Score": f1,
                    "Accuracy": accuracy
                })
                if split_name == "80-20":
                    # Confusion Matrix
                    st.write("Confusion Matrix and ROC Curve for 80-20 Split:")
                    cm = confusion_matrix(Y_test, y_pred_classes)
                    plt.figure(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
                    plt.xlabel('Predicted')
                    plt.ylabel('Actual')
                    plt.title('Confusion Matrix (80-20 Split)')
                    st.pyplot(plt)
                    fpr, tpr, _ = roc_curve(Y_test, y_pred_prob)
                    roc_auc = auc(fpr, tpr)
                    plt.figure(figsize=(8, 6))
                    plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
                    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title('ROC Curve (80-20 Split)')
                    plt.legend(loc="lower right")
                    st.pyplot(plt)
                if split_name == "99-1":
                    st.write("Confusion Matrix and ROC Curve for 99-1 Split:")
                    cm = confusion_matrix(Y_test, y_pred_classes)
                    plt.figure(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
                    plt.xlabel('Predicted')
                    plt.ylabel('Actual')
                    plt.title('Confusion Matrix (99-1 Split)')
                    st.pyplot(plt)
                    fpr, tpr, _ = roc_curve(Y_test, y_pred_prob)
                    roc_auc = auc(fpr, tpr)
                    plt.figure(figsize=(8, 6))
                    plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
                    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title('ROC Curve (99-1 Split)')
                    plt.legend(loc="lower right")
                    st.pyplot(plt)
                if split_name == "75-25":
                    st.write("Confusion Matrix and ROC Curve for 75-25 Split:")
                    cm = confusion_matrix(Y_test, y_pred_classes)
                    plt.figure(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
                    plt.xlabel('Predicted')
                    plt.ylabel('Actual')
                    plt.title('Confusion Matrix (75-25 Split)')
                    st.pyplot(plt)
                    fpr, tpr, _ = roc_curve(Y_test, y_pred_prob)
                    roc_auc = auc(fpr, tpr)
                    plt.figure(figsize=(8, 6))
                    plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
                    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title('ROC Curve (75-25 Split)')
                    plt.legend(loc="lower right")
                    st.pyplot(plt)
                if split_name == "60-40":
                    st.write("Confusion Matrix and ROC Curve for 60-40 Split:")
                    cm = confusion_matrix(Y_test, y_pred_classes)
                    plt.figure(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
                    plt.xlabel('Predicted')
                    plt.ylabel('Actual')
                    plt.title('Confusion Matrix (60-40 Split)')
                    st.pyplot(plt)
                    fpr, tpr, _ = roc_curve(Y_test, y_pred_prob)
                    roc_auc = auc(fpr, tpr)
                    plt.figure(figsize=(8, 6))
                    plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
                    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title('ROC Curve (60-40 Split)')
                    plt.legend(loc="lower right")
                    st.pyplot(plt)
            results_df = pd.DataFrame(results)
            st.write("Evaluation Metrics for Different Splits:")
            st.table(results_df)
            if 'cnn_model' not in st.session_state:
                st.session_state['cnn_model'] = model
                st.session_state['tokenizer'] = tokenizer
            def predict_sentiment(review):
                sequence = st.session_state['tokenizer'].texts_to_sequences([review])
                padded_sequence = pad_sequences(sequence, maxlen=200)
                prediction = st.session_state['cnn_model'].predict(padded_sequence)
                sentiment = "positive" if prediction[0][0] > 0.5 else "negative"
                return sentiment
            user_input = st.text_input("Enter a review:")
            if user_input:
                sentiment = predict_sentiment(user_input)
                st.write(f"The predicted sentiment is: {sentiment}")
        if option == "GateRecurrentUnit":
            data = newdf[['Reviews', 'Sentiment']]
            data.replace({"Sentiment": {"Positive": 1, "Negative": 0}}, inplace=True)
            split_ratios = {
                "80-20": 0.2,
                "99-1": 0.01,
                "75-25": 0.25,
                "60-40": 0.4
            }

            results = []
            for split_name, test_size in split_ratios.items():
                train_data, test_data = train_test_split(data, test_size=test_size, random_state=42)
                tokenizer = Tokenizer(num_words=10000)
                tokenizer.fit_on_texts(train_data["Reviews"])
                X_train = pad_sequences(tokenizer.texts_to_sequences(train_data["Reviews"]), maxlen=200)
                X_test = pad_sequences(tokenizer.texts_to_sequences(test_data["Reviews"]), maxlen=200)
                Y_train = train_data["Sentiment"]
                Y_test = test_data["Sentiment"]
                max_words = 10000
                max_len = 200
                model = Sequential()
                model.add(Embedding(max_words, 128, input_length=max_len))
                model.add(GRU(128, return_sequences=False))
                model.add(Dense(1, activation='sigmoid'))
                model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) 
                model.fit(X_train, Y_train, epochs=5, batch_size=32, validation_split=0.2)
                y_pred_prob = model.predict(X_test).flatten()
                y_pred_classes = (y_pred_prob > 0.5).astype(int)
                precision = precision_score(Y_test, y_pred_classes)
                recall = recall_score(Y_test, y_pred_classes)
                f1 = f1_score(Y_test, y_pred_classes)
                accuracy = accuracy_score(Y_test, y_pred_classes)
                results.append({
                    "Split": split_name,
                    "Precision": precision,
                    "Recall": recall,
                    "F1 Score": f1,
                    "Accuracy": accuracy
                })
                if split_name == "80-20":
                    # Confusion Matrix
                    st.write("Confusion Matrix and ROC Curve for 80-20 Split:")
                    cm = confusion_matrix(Y_test, y_pred_classes)
                    plt.figure(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
                    plt.xlabel('Predicted')
                    plt.ylabel('Actual')
                    plt.title('Confusion Matrix (80-20 Split)')
                    st.pyplot(plt)
                    fpr, tpr, _ = roc_curve(Y_test, y_pred_prob)
                    roc_auc = auc(fpr, tpr)
                    plt.figure(figsize=(8, 6))
                    plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
                    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title('ROC Curve (80-20 Split)')
                    plt.legend(loc="lower right")
                    st.pyplot(plt)
                if split_name == "99-1":
                    st.write("Confusion Matrix and ROC Curve for 99-1 Split:")
                    cm = confusion_matrix(Y_test, y_pred_classes)
                    plt.figure(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
                    plt.xlabel('Predicted')
                    plt.ylabel('Actual')
                    plt.title('Confusion Matrix (99-1 Split)')
                    st.pyplot(plt)
                    fpr, tpr, _ = roc_curve(Y_test, y_pred_prob)
                    roc_auc = auc(fpr, tpr)
                    plt.figure(figsize=(8, 6))
                    plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
                    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title('ROC Curve (99-1 Split)')
                    plt.legend(loc="lower right")
                    st.pyplot(plt)
                if split_name == "75-25":
                    st.write("Confusion Matrix and ROC Curve for 75-25 Split:")
                    cm = confusion_matrix(Y_test, y_pred_classes)
                    plt.figure(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
                    plt.xlabel('Predicted')
                    plt.ylabel('Actual')
                    plt.title('Confusion Matrix (75-25 Split)')
                    st.pyplot(plt)
                    fpr, tpr, _ = roc_curve(Y_test, y_pred_prob)
                    roc_auc = auc(fpr, tpr)
                    plt.figure(figsize=(8, 6))
                    plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
                    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title('ROC Curve (75-25 Split)')
                    plt.legend(loc="lower right")
                    st.pyplot(plt)
                if split_name == "60-40":
                    st.write("Confusion Matrix and ROC Curve for 60-40 Split:")
                    cm = confusion_matrix(Y_test, y_pred_classes)
                    plt.figure(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
                    plt.xlabel('Predicted')
                    plt.ylabel('Actual')
                    plt.title('Confusion Matrix (60-40 Split)')
                    st.pyplot(plt)
                    fpr, tpr, _ = roc_curve(Y_test, y_pred_prob)
                    roc_auc = auc(fpr, tpr)
                    plt.figure(figsize=(8, 6))
                    plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
                    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title('ROC Curve (60-40 Split)')
                    plt.legend(loc="lower right")
                    st.pyplot(plt)
            results_df = pd.DataFrame(results)
            st.write("Evaluation Metrics for Different Splits:")
            st.table(results_df)
            if 'gru_model' not in st.session_state:
                st.session_state['gru_model'] = model
                st.session_state['tokenizer'] = tokenizer
            def predict_sentiment(review):
                sequence = st.session_state['tokenizer'].texts_to_sequences([review])
                padded_sequence = pad_sequences(sequence, maxlen=200)
                prediction = st.session_state['gru_model'].predict(padded_sequence)
                sentiment = "positive" if prediction[0][0] > 0.5 else "negative"
                return sentiment
            user_input = st.text_input("Enter a review:")
            if user_input:
                sentiment = predict_sentiment(user_input)
                st.write(f"The predicted sentiment is: {sentiment}")
        if option == "BidirectionalLSTM":
            data = newdf[['Reviews', 'Sentiment']]
            data.replace({"Sentiment": {"Positive": 1, "Negative": 0}}, inplace=True)
            split_ratios = {
                "80-20": 0.2,
                "99-1": 0.01,
                "75-25": 0.25,
                "60-40": 0.4
            }

            results = []
            for split_name, test_size in split_ratios.items():
                train_data, test_data = train_test_split(data, test_size=test_size, random_state=42)
                tokenizer = Tokenizer(num_words=10000)
                tokenizer.fit_on_texts(train_data["Reviews"])
                X_train = pad_sequences(tokenizer.texts_to_sequences(train_data["Reviews"]), maxlen=200)
                X_test = pad_sequences(tokenizer.texts_to_sequences(test_data["Reviews"]), maxlen=200)
                Y_train = train_data["Sentiment"]
                Y_test = test_data["Sentiment"]
                max_words = 10000
                max_len = 200
                model = Sequential()
                model.add(Embedding(max_words, 128, input_length=max_len))
                model.add(Bidirectional(LSTM(128)))
                model.add(Dense(1, activation='sigmoid'))
                model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                model.fit(X_train, Y_train, epochs=5, batch_size=32, validation_split=0.2)
                y_pred_prob = model.predict(X_test).flatten()
                y_pred_classes = (y_pred_prob > 0.5).astype(int)
                precision = precision_score(Y_test, y_pred_classes)
                recall = recall_score(Y_test, y_pred_classes)
                f1 = f1_score(Y_test, y_pred_classes)
                accuracy = accuracy_score(Y_test, y_pred_classes)
                results.append({
                    "Split": split_name,
                    "Precision": precision,
                    "Recall": recall,
                    "F1 Score": f1,
                    "Accuracy": accuracy
                })
                if split_name == "80-20":
                    # Confusion Matrix
                    st.write("Confusion Matrix and ROC Curve for 80-20 Split:")
                    cm = confusion_matrix(Y_test, y_pred_classes)
                    plt.figure(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
                    plt.xlabel('Predicted')
                    plt.ylabel('Actual')
                    plt.title('Confusion Matrix (80-20 Split)')
                    st.pyplot(plt)
                    fpr, tpr, _ = roc_curve(Y_test, y_pred_prob)
                    roc_auc = auc(fpr, tpr)
                    plt.figure(figsize=(8, 6))
                    plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
                    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title('ROC Curve (80-20 Split)')
                    plt.legend(loc="lower right")
                    st.pyplot(plt)
                if split_name == "99-1":
                    st.write("Confusion Matrix and ROC Curve for 99-1 Split:")
                    cm = confusion_matrix(Y_test, y_pred_classes)
                    plt.figure(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
                    plt.xlabel('Predicted')
                    plt.ylabel('Actual')
                    plt.title('Confusion Matrix (99-1 Split)')
                    st.pyplot(plt)
                    fpr, tpr, _ = roc_curve(Y_test, y_pred_prob)
                    roc_auc = auc(fpr, tpr)
                    plt.figure(figsize=(8, 6))
                    plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
                    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title('ROC Curve (99-1 Split)')
                    plt.legend(loc="lower right")
                    st.pyplot(plt)
                if split_name == "75-25":
                    st.write("Confusion Matrix and ROC Curve for 75-25 Split:")
                    cm = confusion_matrix(Y_test, y_pred_classes)
                    plt.figure(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
                    plt.xlabel('Predicted')
                    plt.ylabel('Actual')
                    plt.title('Confusion Matrix (75-25 Split)')
                    st.pyplot(plt)
                    fpr, tpr, _ = roc_curve(Y_test, y_pred_prob)
                    roc_auc = auc(fpr, tpr)
                    plt.figure(figsize=(8, 6))
                    plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
                    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title('ROC Curve (75-25 Split)')
                    plt.legend(loc="lower right")
                    st.pyplot(plt)
                if split_name == "60-40":
                    st.write("Confusion Matrix and ROC Curve for 60-40 Split:")
                    cm = confusion_matrix(Y_test, y_pred_classes)
                    plt.figure(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
                    plt.xlabel('Predicted')
                    plt.ylabel('Actual')
                    plt.title('Confusion Matrix (60-40 Split)')
                    st.pyplot(plt)
                    fpr, tpr, _ = roc_curve(Y_test, y_pred_prob)
                    roc_auc = auc(fpr, tpr)
                    plt.figure(figsize=(8, 6))
                    plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
                    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title('ROC Curve (60-40 Split)')
                    plt.legend(loc="lower right")
                    st.pyplot(plt)
            results_df = pd.DataFrame(results)
            st.write("Evaluation Metrics for Different Splits:")
            st.table(results_df)
            if 'cnn_model' not in st.session_state:
                st.session_state['cnn_model'] = model
                st.session_state['tokenizer'] = tokenizer
            def predict_sentiment(review):
                sequence = st.session_state['tokenizer'].texts_to_sequences([review])
                padded_sequence = pad_sequences(sequence, maxlen=200)
                prediction = st.session_state['cnn_model'].predict(padded_sequence)
                sentiment = "positive" if prediction[0][0] > 0.5 else "negative"
                return sentiment
            user_input = st.text_input("Enter a review:")
            if user_input:
                sentiment = predict_sentiment(user_input)
                st.write(f"The predicted sentiment is: {sentiment}")
    if selected == "Hugging Face Transformer":
        st.write("Note: These group of models will not undergo training as it is a pre-trained model only output will be generated")
        option = st.selectbox(
        "Select a Model:",
        ["mrm8488/t5-base-finetuned-emotion", "j-hartmann/emotion-english-distilroberta-base", "michellejieli/emotion_text_classifier", "LiYuan/amazon-review-sentiment-analysis", "cardiffnlp/twitter-roberta-base-sentiment-latest"]
        )
        if option == "mrm8488/t5-base-finetuned-emotion":
           user_input = st.text_input("Enter the review")
           pipe = pipeline("text2text-generation", model="mrm8488/t5-base-finetuned-emotion")
           sentiment=pipe(user_input)
           st.write("Model Output")
           st.write(sentiment[0]['generated_text'])
        if option == "j-hartmann/emotion-english-distilroberta-base":
            user_input = st.text_input("Enter the review")
            classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")
            sentiment1=classifier(user_input)
            st.write("Model Output")
            st.write(sentiment1[0]['label'])   
            st.write(sentiment1[0]['score'])
        if option == "michellejieli/emotion_text_classifier":
            user_input = st.text_input("Enter the review")
            classifier = pipeline("sentiment-analysis", model="michellejieli/emotion_text_classifier")
            sentiment2=classifier(user_input)
            st.write("Model Output")
            st.write(sentiment2[0]['label'])   
            st.write(sentiment2[0]['score'])
        if option == "LiYuan/amazon-review-sentiment-analysis":
            user_input = st.text_input("Enter the review")
            classifier = pipeline("text-classification", model="LiYuan/amazon-review-sentiment-analysis")
            sentiment3=classifier(user_input)
            st.write("Model Output")
            st.write(sentiment3[0]['label'])   
            st.write(sentiment3[0]['score'])
        if option == "cardiffnlp/twitter-roberta-base-sentiment-latest":
            user_input = st.text_input("Enter the review")
            classifier = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
            sentiment4=classifier(user_input)
            st.write("Model Output")
            st.write(sentiment4[0]['label'])   
            st.write(sentiment4[0]['score'])
        
                                                                                             
