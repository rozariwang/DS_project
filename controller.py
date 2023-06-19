import openai
import random
import streamlit as st

''' :)
# Retrieve OpenAI API credentials from secrets.toml
secrets = st.secrets["openai"]
openai_organization = secrets["openai_organization"]
openai_api = secrets["openai_api"]
'''   

# Set up OpenAI API credentials
#openai.organization = openai_organization
openai.organization = "org-pJcWPQGFUTRBlstxxYtLSgys"
#openai.api_key = openai_api
openai.api_key = "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"

# Function to generate recommendation using ChatGPT API
@st.cache
def generate_recommendation(company):
    score = generate_random_prediction()
    
    prompt = f"Given the score -1(not promising) to 1(very promosing), our model gives {company} company a score of {score}. Based on this score {score}, explain why {company} company stock might or might not be suitable to invest in the long term? Respond in less than 150 words."
    completion = openai.Completion.create(
        engine='text-davinci-003',
        prompt=prompt,
        max_tokens=1000,
        n=1,
        stop=None,
        temperature=0.6
    )
    response = completion.choices[0].text
    return response

def generate_random_prediction():
    return random.randint(-1, 1)
