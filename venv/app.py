import streamlit as st
import pandas as pd

st.title("Retail Media Incrementality Engine")
st.subheader("Scenario 3: Independent Platform")

uploaded_file = st.file_uploader("Upload your Retail Media Data (CSV)", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview:", df.head())
    st.success("Engine Ready: Bayesian Causal Inference can now be applied.")