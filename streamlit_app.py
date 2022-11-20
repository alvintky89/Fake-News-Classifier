import streamlit as st
import requests
import json
from PIL import Image
import base64

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('background4.jpeg') 

# Title of the webpage
st.title("Fake News Classifier")

st.subheader("""Use this application to differentiate REAL and FAKE news.""")

with st.form(key='myform', clear_on_submit = True):
    article = st.text_input('Enter the text of the news')
    submit = st.form_submit_button("Read news")

user_input = {'article': article}

st.write(user_input)
if submit:
    with st.spinner('Yes, reading news...'):
        api_url = 'http://localhost:8080' # specify the URL to access
        api_route = '/predict' # specify the `route` to access in the URL
        
        # we'll need to use `requests.post()` based on our earlier specification in `\predict` route to only accept a `POST` request 
        response = requests.post(f'{api_url}{api_route}', json=json.dumps(user_input))
        predictions = response.json()
        
        st.success('Finished Reading')
        st.header('Verdict:')
        st.write(predictions['prediction'])
        #st.write(f"Prediction: {predictions['predictions'][0]}")
