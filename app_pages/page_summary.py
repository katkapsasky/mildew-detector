import streamlit as st
import matplotlib.pyplot as plt


def page_summary_body():

    st.write("### Quick Project Summary")

    st.info(
        f"**General Information**")

    st.write(
        f"* For additional information, please visit and **read** the "
        f"[Project README file](https://github.com/katkapsasky/mildew-detector/blob/main/README.md).")  # noqa

    st.success(
        f"The project has 2 business requirements:\n"
        f"* 1 - The client is interested in conducting a study to visually "
        f"differentiate a cherry leaf that is healthy from one that "
        f"contains powdery mildew..\n"
        f"* 2 - The client is interested in predicting if a cherry leaf is "
        f"healthy or contains powdery mildew. "
    )
