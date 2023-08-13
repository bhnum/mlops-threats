import streamlit as st
import requests as rs

text = st.text_input("Text")


def get_api(params):
    url = "http://api:8086/predict/"
    response = rs.get(url, params=params, timeout=20)
    return response.content


if st.button("Get response"):
    params = {
        "text": text,
    }

    data = get_api(params)
    st.write(data)
