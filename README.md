# VAVAVOICE : Speech Emotion Recognition with Wav2Vec2 and Emotion Classification

This repository contains code for training and using ASR and Speech Emotion Recognition (SER) system using Wav2Vec2 for feature extraction and a fine-tuned Emotion Classification model.

## Table of Contents

- [Introduction](#introduction)
- [Setup](#setup)

## Introduction

Automatic Speech Recognition is the task of transcribing a raw audio into text. Speech Emotion Recognition is the task of automatically recognizing emotions from speech signals. In this project, we train 2 model from scratch for ASR using a simple MLP and then a bi-GRU model and then we use the Wav2Vec2 model for feature extraction from audio signals and fine-tune an Emotion Classification model to predict the emotion from the extracted features.

## Setup

1. **Install Dependencies**: Make sure you have all the required dependencies installed. You can install them using the following command:

    ```bash
    pip install -r requirements.txt
    ```


## Streamlit App

This repository includes a small Streamlit web application for real-time Automatic speech recognition and speech emotion recognition. Follow the steps below to run the app:

1. **Install Streamlit**: If you haven't installed Streamlit, you can do so by running:

    ```bash
    pip install streamlit
    ```

2. **Run the Streamlit App**: Execute the following command to run the Streamlit app locally:

    ```bash
    streamlit run app.py
    ```

3. **Open the App in a Browser**: After running the command, a new tab will open in your default web browser displaying the app. You can use your microphone to make real-time predictions.

4. **Interact with the App**: Follow the on-screen instructions to interact with the app.

Note: Ensure that your environment is set up correctly, including all dependencies, before running the Streamlit app. Refer to the [Setup](#setup) section for more information.

