import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.image import imread
from src.machine_learning.evaluate_clf import load_test_evaluation


def page_ml_performance_metrics():
    version = 'v1'

    st.write("### Label Distribution Across Train, Validation and Test Sets")
    st.success(
        f"The bar and pie charts below provide a visual display of how the "
        f"dataset was split in preparation for fitting and training the "
        f"image classification model."
    )

    labels_distribution = plt.imread(
        f"outputs/{version}/labels_distribution.png"
    )
    st.image(labels_distribution, caption='Labels Distribution - Bar Chart')

    labels_distribution = plt.imread(
        f"outputs/{version}/labels_distribution_pie.png"
    )
    st.image(labels_distribution, caption='Labels Distribution - Pie Chart')

    st.success(
        f"Following convention, 70% of the data was used to train the model, "
        f"10% was used for validation and the remaining 20% was reserved "
        f"to evaluate the model after training."
    )
    st.write("---")

    st.write("### Model Confusion Matrix")
    st.success(
        f"The Confusion Matrix allows us to summarize the performance of the "
        f"classification model by displaying the sum of accurately and "
        f"inaccurately predicted leaves for both healthy and infected labels."
    )

    model_confusion = plt.imread(f"outputs/{version}/confusion_matrix.png")
    st.image(model_confusion, caption='Confusion Matrix')

    st.write("### Model History")
    st.success(
        f"By plotting the learning curves for the model's training and "
        f"accuracy on the train and validation data we are able to detect if "
        f"the model has trained normally, is overfitting "
        f"(memorising the data) or underfitting "
        f"(not learning the data at all)."
    )

    col1, col2 = st.beta_columns(2)
    with col1:
        model_acc = plt.imread(f"outputs/{version}/model_training_acc.png")
        st.image(model_acc, caption='Model Training Accuracy')
    with col2:
        model_loss = plt.imread(f"outputs/{version}/model_training_losses.png")
        st.image(model_loss, caption='Model Training Losses')
    st.write("---")

    st.write("### Generalised Performance on Test Set")
    st.dataframe(pd.DataFrame(load_test_evaluation(
        version), index=['Loss', 'Accuracy']))
