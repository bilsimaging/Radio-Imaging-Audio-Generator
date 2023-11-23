import streamlit as st
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import scipy.io.wavfile
import openai
import time  # Used for simulating progress
import torch
import tensorflow as tf


# Streamlit app setup
st.set_page_config(
    page_icon="https://soundboard.bilsimaging.com/faviconbilsimaging.png", 
    layout="wide",
    page_title='Radio Imaging Audio Generator Beta 0.1',
    initial_sidebar_state="expanded"
)

# Main Description and Header
st.markdown("""
    <h1 style=''>Radio Imaging Audio Generator 
    <span style='font-size: 24px; color: #FDC74A;'>Beta 0.1</span></h1>
    """, unsafe_allow_html=True)
st.write("Welcome to the Radio Imaging & MusicGen Ai audio generator. This web application allows you to easily create unique audio for your radio imaging projects or any music creators using AI technology.")
st.markdown("---")

# How to Use the App - Instructions
with st.expander('How to start use the WEB APP ? 📌'):
    st.write('''
        1. **Enter OpenAI API Key:** In the sidebar, input your OpenAI API key. This key is required to access the GPT model for generating audio descriptions.
           - If you don’t have an OpenAI API key, you can obtain one for free [here](https://platform.openai.com/account/api-keys).
        2. **Select GPT Model:** Choose the GPT model from the dropdown in the sidebar. The 'gpt-3.5-turbo-16k' model provides more detailed descriptions.
        3. **Input Your Prompt:** In the text area, enter a detailed description of the audio piece you want to create.
        4. **Generate Audio:** Click 'Generate Audio' to process your request.
        5. **Playback and Download:** Once the audio is generated, play it directly within the app.
    ''')

# Sidebar for user inputs
with st.sidebar:
    openai_api_key = st.text_input("OpenAI API key", type="password", help="Enter your OpenAI API key here.")
    st.caption("*If you don't have an OpenAI API key, get it [here](https://platform.openai.com/account/api-keys).*")
    model = st.selectbox("OpenAI chat model", ("gpt-3.5-turbo", "gpt-3.5-turbo-16k"), help="Select the desired GPT model.")
    st.markdown("Check out our video tutorials on [YouTube](https://www.youtube.com/channel/UCdDH7T3oa8YMPFV5e79skaA) for helpful guides on using this app!")
    st.markdown('''Made with ❤️ by [Bilsimaging](https://bilsimaging.com)''', unsafe_allow_html=True)   

# Prompt input
st.markdown("## ✍🏻Prompt input")
prompt = st.text_area("Enter your radio imaging prompt here", help="Describe the audio piece you want to create.")

# Instructions for users
st.info("Provide a detailed description of the audio you need, such as mood, instruments, and style. Example: A calm, soothing melody with soft piano and nature sounds for a morning show.")

# Generate Audio Button with Progress Bar
if st.button("Generate Audio"):
    if not openai_api_key.strip() or not prompt.strip():
        st.error("Please provide both the OpenAI API key and a description for your radio imaging.")
    else:
        with st.spinner("Generating your audio..."):
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.1)  # Simulate a task
                progress_bar.progress(i + 1)
            
            try:
                full_prompt = {"role": "user", "content": f"Describe a radio imaging audio piece based on: {prompt}"}
                response = openai.ChatCompletion.create(model=model, messages=[full_prompt], api_key=openai_api_key)
                descriptive_text = response.choices[0].message['content'].strip()

                processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
                musicgen_model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
                inputs = processor(text=[descriptive_text], padding=True, return_tensors="pt")
                audio_values = musicgen_model.generate(**inputs, max_new_tokens=512)
                sampling_rate = musicgen_model.config.audio_encoder.sampling_rate

                audio_filename = "radio_imaging_output.wav"
                scipy.io.wavfile.write(audio_filename, rate=sampling_rate, data=audio_values[0, 0].numpy())
                st.success("Your audio has been successfully created! Below is a description of your audio piece based on the GPT model's understanding:")
                st.write(descriptive_text)
                st.audio(audio_filename)
            except Exception as e:
                st.error(f"An error occurred: {e}")
            finally:
                progress_bar.empty()  # Remove the progress bar after completion

# Footer and Support Section
st.markdown("---")
st.markdown("## 🌐 Project Continuation and User Involvement")
st.markdown("🔊 This app is the next step in our project, following the Custom GPTs Radio Imaging and MusicGen AI. <br>It's tailored for radio producers and music creators, offering new levels of creativity and efficiency by Bilsimaging. [Try our GPTs](https://chat.openai.com/g/g-65x53n87E-radio-imaging-musicgen-ai).", unsafe_allow_html=True)
st.markdown("If you appreciate my deployment and wish to support me, please consider a donation. Your support helps me continue providing value. Thank you for joining me on this journey! - Bilel Aroua")
st.markdown("For support ☕ [Buy me a Coffee](https://ko-fi.com/bilsimaging).")
st.image('https://storage.ko-fi.com/cdn/brandasset/kofi_button_dark.png', width=300, caption="Project Bilsimaigng")

# Hide Streamlit branding
st.markdown("<style>#MainMenu {visibility: hidden;} footer {visibility: hidden;}</style>", unsafe_allow_html=True)
