import streamlit as st
from ui.pages.index import UI

# Create an instance of the UI class
ui = UI()

# Init pages
ui.setup_pages(show_title=False)

st.info("Please choose the tab to start!", icon="ℹ️")
