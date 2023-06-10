import streamlit as st
import matplotlib.pyplot as plt


def page_summary_body():

    st.write("### Quick Project Summary")

    st.info(
        f"**General Information**\n\n"
        f"Powdery mildew is a fungal disease of the foliage and "
        f"stems of a plant, where a superficial fungal growth covers the "
        f"surface of the plant.\n"
        f"The disease can be visually identified by white, powdery "
        f"spreading patches of fungus on a leaf's surface.\n\n"
        f"Leaves infected with powdery mildew may gradually turn completely "
        "yellow, die, and fall off, which can expose fruit to sunburn. "
        f"Severely infected plants can result in reduced yields, shortened "
        f"production times, and fruit that has little flavor.")

    st.info(
        f"**Dataset Content**\n\n"
        f"The dataset, provided by the client, contains 4208 images of cherry "
        f"leaves; 2104 healthy and 2104 powdery mildew infected leaves. ")

    st.write(
        f"* For additional information, please visit and **read** the "
        f"[Project README file](https://github.com/katkapsasky/mildew-detector/blob/main/README.md).")  # noqa

    st.success(
        f"The project has 2 business requirements:\n"
        f"* 1 - The client is interested in conducting a study to visually "
        f"differentiate a cherry leaf that is healthy from one that "
        f"contains powdery mildew.\n\n"
        f"* 2 - The client is interested in predicting if a cherry leaf is "
        f"healthy or contains powdery mildew. "
    )
