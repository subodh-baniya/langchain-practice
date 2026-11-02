import streamlit as st


st.set_page_config(page_title="LLM Streamlit App", layout="wide")
st.title("LLM Streamlit App")

st.sidebar.title("Sidebar")

st.sidebar.markdown(
    """
    This is a simple Streamlit app that demonstrates the use of a large language model (LLM) for various tasks. You can use the sidebar to navigate through different functionalities.
    """ )

st.button("Click me!", on_click=lambda: st.write("Button clicked!"))