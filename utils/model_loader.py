import pickle
import os
import streamlit as st

def load_model(model_path):
    """Loads a pre-trained model from a .pkl file."""
    if not os.path.exists(model_path):
        st.error(f"Error: Model file not found at {model_path}") # Use st.error for Streamlit display
        return None
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model
