import streamlit as st


def about_page():
    st.title("Project Description 🏋️‍♂️")
    st.header("Mission 🎯")
    st.write(
        "Our mission is to explore the feasibility of reproducing a specified curve using measurements from a LIDAR, "
        "a 3D camera, or other measurement devices. We are also seeking to determine whether adding image analysis is "
        "necessary to enhance the accuracy of the reproduction.")

    st.header("Objectives 🚀")
    st.write("1. **Equipment Design:** Develop a device capable of reproducing the specified curve.")
    st.write(
        "2. **Sensor Evaluation:** Determine if measuring with a 3D camera (or possibly other sensors) alone is "
        "sufficient or if image analysis needs to be integrated.")
    st.write(
        "3. **Precision Optimization:** In case image analysis is necessary, work on methods to optimize the "
        "precision of the reproduction.")

    st.header("Note ℹ️")
    st.write("This project is part of the 'PIE - Projet d'Ingénieur en Équipe' course at ENSTA Paris.")


about_page()
