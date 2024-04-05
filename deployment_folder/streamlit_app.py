import streamlit as st
import requests
from inference import *

# Streamlit app logic
def main():
    # Add an image to the page
    st.image("/Users/kailashkumar/Documents/week_5_transformers/transformer_image.jpg", use_column_width=True)

    # Add input for start word
    start_word = st.text_input("Enter a start word for story generation: (careful what you type for!)")

    # Center-align the button
    st.markdown(
        """
        <style>
        .stButton>button {
            display: block;
            margin: 0 auto;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    if st.button("Generate Story"):
        story = generate_text(start_word)
        st.header("Story:")
        st.write(f"{story}")

if __name__ == "__main__":
    main()
