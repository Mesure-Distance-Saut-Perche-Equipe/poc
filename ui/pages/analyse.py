import streamlit as st
from PIL import Image


def main():
    st.title('Pole Vault Distance Measurement 🏋️‍')
    st.subheader('Welcome to the project dedicated to the measuring distance to bar for pole vault')

    # Use the random key when creating the file uploader
    file = st.file_uploader('Choose a file')
    if file is not None:
        st.title("Here is the image you've uploaded")
        img = Image.open(file)
        print(img)
        st.image(img)


main()
