import streamlit as st
import requests as r
from os.path import dirname, join, realpath
import joblib
from langdetect import detect

# add banner image
st.header("Swahili Statements Sentiment Analysis App")
st.image("Image/emojis.jpg")
st.subheader(
    """
A simple app to analyze the sentiment of Swahili statements.
"""
)

# form to collect news content
my_form = st.form(key="news_form")
tweet = my_form.text_input("Input your swahili statement here")
submit = my_form.form_submit_button(label="make prediction")


# load the model and count_vectorizer

with open(
    join(dirname(realpath(__file__)), "Model/xg_ClassifierModel.pkl"), "rb"
) as f:
    model = joblib.load(f)

with open(join(dirname(realpath(__file__)), "Preprocessing/vectorizer_tfidf.pkl"), "rb") as f:
    vectorizer = joblib.load(f)

sentiments = {0: "Neutral", 1: "Positive", 2: "Negative"}


if submit:

    if detect(tweet) == "sw":

        # transform the input
        transformed_tweet = vectorizer.transform([tweet])
        #transformed_tweet =  transformed_tweet.toarray()

        # perform prediction
        prediction = model.predict(transformed_tweet)
        output = int(prediction[0])
        probas = model.predict_proba(transformed_tweet)
        probability = "{:.2f}".format(float(probas[:, output]))

        # Display results of the NLP task
        st.header("Results")
        if output == 1:
            st.write("The sentiment of the statement is {} 😊 with the probability of {}".format(sentiments[output],probability))
        elif output == -1:
            st.write("The sentiment of the statement is {} 😡 with the probability of {}".format(sentiments[output],probability))
        else:
            st.write("The sentiment of the statement is {} 😐 with the probability of {}".format(sentiments[output],probability))

    else:
        st.write(" ⚠️ The tweet is not in swahili language.Please make sure the input is in swahili language")

