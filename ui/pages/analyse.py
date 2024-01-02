import streamlit as st
from PIL import Image
from analyse.model import image_processor


def main():
    st.title('Pole Vault Distance Measurement 🏋️‍')
    st.subheader('Welcome to the project dedicated to the measuring distance to bar for pole vault')

    # Use the random key when creating the file uploader
    file = st.file_uploader("Choose a file", type=["png", "jpg", "jpeg"])
    if file is not None:
        st.title("Here is the image you've uploaded")
        img = Image.open(file)
        st.image(img)

        with st.spinner("Analysing..."):
            resulted_image, results = image_processor.process(img)

            # Show results
            st.title("Here is the result of analyse")

            # Nothing is detected
            if len(results) == 0:
                st.subheader("No objects were detected")
            else:
                # Display image with the detected objects
                st.subheader("Processed image")
                st.image(resulted_image)

                # Print information about the detection
                st.subheader("Detected objects")
                for _, (score, label, box) in enumerate(results):
                    st.markdown(f"**{label}** with confidence **{round(score.item(), 3)}** at location **{box}**")


main()
