import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.image import imread
from src.machine_learning.evaluate_clf import load_test_evaluation


def page_ml_performance_metrics():
    version = 'v1'

    st.write("### Label Distribution Across Train, Validation and Test Sets")
    labels_distribution = plt.imread(
        f"outputs/{version}/labels_distribution.png"
    )
    st.image(labels_distribution, caption='Labels Distribution - Bar Chart')

    labels_distribution = plt.imread(
        f"outputs/{version}/labels_distribution_pie.png"
    )
    st.image(labels_distribution, caption='Labels Distribution - Pie Chart')

    st.write("---")

    st.write("### Model Confusion Matrix")

    model_confusion = plt.imread(f"outputs/{version}/confusion_matrix.png")
    st.image(model_confusion, caption='Confusion Matrix')

    st.write("### Model History")
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
