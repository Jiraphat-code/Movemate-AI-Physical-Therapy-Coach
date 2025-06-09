import streamlit as st
from streamlit_option_menu import option_menu
import os
from dotenv import load_dotenv

# ✅ Must be the very first Streamlit command
st.set_page_config(
    page_title="MoveMate",
)

# ✅ Now it's safe to load other modules
import pages.home as home
import pages.register as register
import pages.feature_select as feature_selection
import pages.processing as processing_results

class MultiApp:

    def __init__(self):
        self.apps = []

    def add_app(self, title, func):
        self.apps.append({"title": title, "function": func})

    def run(self):
        with st.sidebar:
            app = option_menu(
                menu_title='MoveMate',
                options=['Home', 'Register', 'Feature Selection', 'Processing Results'],
                icons=['house-door-fill', 'person-plus-fill', 'clipboard2-pulse-fill', 'graph-up'],
                menu_icon='chat-text-fill',
                default_index=0,
                styles={
                    "container": {"padding": "5!important", "background-color": 'black'},
                    "icon": {"color": "white", "font-size": "23px"},
                    "nav-link": {
                        "color": "white", "font-size": "20px", "text-align": "left",
                        "margin": "0px", "--hover-color": "blue"
                    },
                    "nav-link-selected": {"background-color": "#02ab21"},
                }
            )

        if app == "Home":
            home.app()
        if app == "Register":
            register.app()
        if app =="Feature Selection":
            feature_selection.app()
        if app == "Processing Results":
            processing_results.app()


# ✅ Run the multi-app structure
if __name__ == '__main__':
    app = MultiApp()
    app.run()
