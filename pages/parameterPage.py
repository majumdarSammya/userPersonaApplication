import streamlit as st
from PIL import Image

st.markdown("# Parameter Page")
st.sidebar.markdown("# Parameter Page")

column1, column2 = st.columns(2)

with column1:
    st.header("Parameters")

    weightParameter = st.text_input('Weight')
    st.write(weightParameter)

    ageParameter = st.text_input('Age')
    st.write(ageParameter)

    genderParameter = st.selectbox('Gender:', ('Female','Male'))
    st.write("You selected:", genderParameter)

    st.button("Tune Results")

with column2:
    st.header("Results")

    inputBox = st.text_input('Insert Prompt:')
    st.write(inputBox)

    image = Image.open('stickFigure.png')
    st.image(image, caption='Persona')

    st.button("Export")
