# Import packages and libraries
import streamlit as st
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.image import imread

import itertools
import random

def page_leaf_visualizer_body():
    st.write("### Cherry Leaf Visualizer")
    st.info(
        f"The client is interested in conducting a study to visually "
        f"differentiate a cherry leaf that is healthy from one that contains "
        f"powdery mildew.")
    
    version = 'v1'
    if st.checkbox("Difference between average and variability image"):
      
      avg_parasitized = plt.imread(f"outputs/{version}/avg_var_healthy.png")
      avg_uninfected = plt.imread(
        f"outputs/{version}/avg_var_powdery_mildew.png"
        )

      st.warning(
        f"* We notice the average and variability images didn't show "
        f"patterns where we could intuitively differentiate one to another." 
        f"However, leaves infected with mildew do present lighter / white "
        f"stripes across them, which isn't visible in healthy leaves.")

      st.image(avg_healthy, caption='Healthy Leaf - Average & Variability')
      st.image(avg_infected, caption='Infected Leaf - Average &Variability')
      st.write("---")

    if st.checkbox("Differences between average healthy and infected leaves"):
          diff_between_avgs = plt.imread(f"outputs/{version}/avg_diff.png")

          st.warning(
            f"* We notice this study didn't show "
            f"patterns where we could intuitively differentiate " 
            f"one to another.")
          st.image(
            diff_between_avgs, caption='Difference between average images'
            )

    if st.checkbox("Image Montage"): 
      st.write("* To refresh the montage, click on the 'Create Montage' button")
      my_data_dir = 'inputs/cherry_leaves_dataset/cherry-leaves'
      labels = os.listdir(my_data_dir+ '/validation')
      label_to_display = st.selectbox(
        label="Select label", options=labels, index=0
        )
      if st.button("Create Montage"):      
        image_montage(dir_path= my_data_dir + '/validation',
                      label_to_display=label_to_display,
                      nrows=8, ncols=3, figsize=(10,25))
      st.write("---")
