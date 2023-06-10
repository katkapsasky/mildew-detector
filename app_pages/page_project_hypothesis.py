# Import packages
import streamlit as st
import matplotlib.pyplot as plt


def page_project_hypothesis_body():
    st.write("### Project Hypotheses and Validation")

    st.success(
        f"**Hypothesis 1**\n\n"
        f"We suspect cherry leaves infected with powdery mildew have clear "
        f"marks/signs, that can differentiate, from a healthy leaf."
    )

    st.warning(
        f"**Hypothesis 1 - Validation**\n\n"
        f"By computing the average images for healthy and powdery mildew "
        f"infected cherry leaves and displaying a comparison, we can visually "
        f"check for differences in the leaves.\n\n"
        f"Creating an image montage provides further visualisation of "
        f"healthy and infected cherry leaves.\n\n"
        f"Although images in the montage and the average variability and "
        f"difference of the two images do present visual distinctions - "
        f"infected leaves have more white stripes across the center of the "
        f"leaf whereas healthy ones are a more opaque green - when plotting "
        f"the difference image of the two we notice no patterns where we "
        f"could intuitively differentiate one from another.\n\n"
        f"**For more information on the model, visit the Leaf Visualizer "
        f"page.**"
    )

    st.success(
        f"**Hypothesis 2**\n\n"
        f"We suspect that an image classification model can be used to "
        f"predict whether a leaf in a given image is healthy or infected with "
        f"powdery mildew."
    )

    st.warning(
        f"**Hypothesis 2 - Validation**\n\n"
        f"By training the image classification model using multi-class "
        f"classification and a portion of the dataset we are able to predict "
        f"the health of one or more leaf images at a time with 93% accuracy "
        f"on the tested data set.\n\n"
        f"**For more information on the model, visit the ML Performance "
        f"Metrics page.**"
    )

    st.success(
        f"**Hypothesis 3**\n\n"
        f"We suspect that reducing the size of the images in the dataset "
        f"will allow the model to train faster without compromising the "
        f"accuracy of prediction to an unsatisfactory level."
    )

    st.warning(
        f"**Hypothesis 3 - Validation**\n\n"
        f"Resizing the images in the dataset from 256 x 256 pixels to "
        f"100 x 100 pixels allowed the model to train over 80% faster, "
        f"completing each epoch in approximately 40 seconds as opposed to "
        f"approximately 240 seconds with the original image size. However, "
        f"the quality of the images was compromised and the model was "
        f"overfitting, which hinders the accuracy of prediction on unseen "
        f"data. Due to this, the model was trained with images at the "
        f"original S256 x 256 pixels size."
    )
