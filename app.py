import streamlit as st
import sounddevice as sd
import numpy as np
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from transformers import Wav2Vec2ForSequenceClassification

MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-french"
processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
transcription_model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)

emotion_model = Wav2Vec2ForSequenceClassification.from_pretrained("Djulo/wav2vec2-emotion_recognition_model_fr")

emotion_map = {
    0: "Anger",
    1: "Disgust",
    2: "Fear",
    3: "Happiness",
    4: "Sadness",
    5: "Surprise"
}

def transcribe(raw_speech):
    inputs = processor(raw_speech, sampling_rate=16_000, return_tensors="pt", padding=True)

    with torch.no_grad():
        transcription_model_logits = transcription_model(inputs.input_values.float(), attention_mask=inputs.attention_mask).logits
        emotion_model_logits = emotion_model(inputs.input_values.float(), attention_mask=inputs.attention_mask).logits

    # Predict Transcribe
    predicted_ids_tr = torch.argmax(transcription_model_logits, dim=-1)
    predicted_sentences_tr = processor.batch_decode(predicted_ids_tr)

    # Predict emotion
    predicted_ids_emt = torch.argmax(emotion_model_logits, dim=-1).detach().cpu().numpy()

    return predicted_sentences_tr, predicted_ids_emt

# Function to record audio
def record_audio(seconds=5, sample_rate=44100):
    audio_data = sd.rec(int(seconds * sample_rate), samplerate=sample_rate, channels=1, dtype=np.int16)
    sd.wait()
    return audio_data.flatten()

def main():
    st.title("Voice Recognition And Emotion App")

    recording = False

    if not recording and st.button("Start Recording"):
        recording = True
        st.info("Recording... Speak into the microphone.")
        audio_data = record_audio()
        st.success("Recording complete!")

        transcription, emotion = transcribe(audio_data)

        st.subheader("Recognized Text:")
        st.text(transcription[0])
        st.text(f"Emotion: {emotion_map[emotion[0]]}")

if __name__ == "__main__":
    main()
