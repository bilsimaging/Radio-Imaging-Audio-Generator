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
with st.expander('📘 How to Use This Web App?'):
    st.markdown('''
        To get started with creating your unique audio pieces, follow these simple steps:

        **1. Enter OpenAI API Key**
        - In the sidebar, input your **OpenAI API key**. This is essential to access the GPT model for generating audio descriptions.
        - Don't have an API key? Get one for free [here](https://platform.openai.com/account/api-keys).

        **2. Select GPT Model**
        - Choose the desired GPT model from the dropdown in the sidebar. We recommend using **'gpt-3.5-turbo-16k'** for more detailed and rich descriptions.

        **3. Input Your Detailed Description**
        - Describe your audio idea in the text area provided. Be as detailed as possible to guide the AI effectively. This could include the mood, style, specific instruments, or any other relevant details.

        **4. Generate and Review the Prompt**
        - Click on **'📄 Generate Prompt'** to create a descriptive prompt for your audio. Review it to ensure it aligns with your vision.

        **5. Generate Your Audio**
        - If you're satisfied with the prompt, hit **'▶ Generate Audio'**. This will process your request and create the audio piece based on the AI-generated description.

        **6. Playback and Download**
        - Once generated, you can play the audio directly within the app. If it meets your needs, feel free to download and use it in your projects.
    ''')


# Sidebar for user inputs
with st.sidebar:
    openai_api_key = st.text_input("OpenAI API key", type="password", help="Enter your OpenAI API key here.")
    st.caption("*If you don't have an OpenAI API key, get it [here](https://platform.openai.com/account/api-keys).*")
    model = st.selectbox("OpenAI chat model", ("gpt-3.5-turbo", "gpt-3.5-turbo-16k"), help="Select the desired GPT model.")
    st.markdown("Check out our video tutorials on [YouTube](https://www.youtube.com/channel/UCdDH7T3oa8YMPFV5e79skaA) for helpful guides on using this app!")
    st.markdown('''Made with ❤️ by [Bilsimaging](https://bilsimaging.com)''', unsafe_allow_html=True)   

# Guidelines for generating prompt and audio
st.markdown("""
    ### 💡 Steps to Generate Your Audio:
    1. **Write your Detailed Description**   
    2. **Generate Prompt**: Click 'Generate Prompt' to create a description for your radio imaging audio.
    3. **Review the Prompt**: Read the output and make sure it aligns with what you have in mind.
    4. **Generate Audio**: If you are satisfied with the prompt, click 'Generate Audio' to create your audio piece.
""")

# Prompt input
st.markdown("## ✍🏻Write your Description")
prompt = st.text_area("Enter your radio imaging draft idea prompt here", help="Describe the audio piece you want to create.")

# Instructions for users
st.info("👉🏻 Provide a detailed description of the audio you need, such as mood, instruments, and style. Example: A calm, soothing melody with soft piano for a morning show.")

# Generate Prompt Button with user confirmation and patience message
st.markdown("## 📝 Generate Prompt")
st.info("🚨 Generating the prompt may take a few moments. Please be patient.")
if st.button("📄 Generate Prompt"):
    if not openai_api_key.strip() or not prompt.strip():
        st.error("Please provide both the OpenAI API key and a description for your radio imaging.")
    else:
        with st.spinner("Generating your prompt... Please wait, this might take a few moments."):
            try:
                full_prompt = {"role": "user", "content": f"Describe a radio imaging audio piece based on: {prompt}"}
                response = openai.ChatCompletion.create(model=model, messages=[full_prompt], api_key=openai_api_key)
                descriptive_text = response.choices[0].message['content'].strip()

                # Append a copyright notice or tag
                copyright_notice = "\n\n© Created through Radio Imaging Audio Generator by Bilsimaging [WEBSITE](https://bilsimaging.com)"
                descriptive_text += copyright_notice

                st.session_state['generated_prompt'] = descriptive_text
                st.success("Your prompt has been successfully generated! Review the prompt below:")
                st.write(descriptive_text)

                # Download Button for the generated prompt
                st.download_button(
                    label="Download Prompt",
                    data=descriptive_text,
                    file_name="generated_prompt.txt",
                    mime="text/plain"
                )

            except Exception as e:
                st.error(f"An error occurred: {e}")

st.markdown("---")



# Generate Audio Button with Progress Bar and Load Management
st.markdown("## 🎶 Generate Audio")
st.info("🚨 Please be patient as generating audio can take some time. This might take a moment due to resource limits. Feel free to notify me if you encounter any issues.")   

if st.button("▶ Generate Audio"):
    if 'generated_prompt' not in st.session_state or not st.session_state['generated_prompt']:
        st.error("Please generate and approve a prompt before creating audio.")
    else:
        descriptive_text = st.session_state['generated_prompt']
        
        # Placeholder for server load check
        server_ready_for_audio_generation = True  # Replace with actual server load check logic

        if server_ready_for_audio_generation:
            with st.spinner("Generating your audio... Please wait, this might take a few moments."):
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.1)  # Simulate processing
                    progress_bar.progress(i + 1)
                
                try:
                    processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
                    musicgen_model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
                    inputs = processor(text=[descriptive_text], padding=True, return_tensors="pt")
                    audio_values = musicgen_model.generate(**inputs, max_new_tokens=512)
                    sampling_rate = musicgen_model.config.audio_encoder.sampling_rate

                    audio_filename = "Bilsimaging_radio_imaging_output.wav"
                    scipy.io.wavfile.write(audio_filename, rate=sampling_rate, data=audio_values[0, 0].numpy())
                    st.success("Your audio has been successfully created! Below is a description of your audio piece based on the GPT model's understanding:")
                    st.write(descriptive_text)
                    st.audio(audio_filename)
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                finally:
                    progress_bar.empty()  # Remove the progress bar after completion
        else:
            st.warning("The server is currently busy. Please try generating your audio again later.")



# Footer and Support Section
st.markdown("---")
st.markdown("## 🌐 Project Continuation and User Involvement")
st.markdown("✔️ This app is the next step in our project, following the Custom GPTs Radio Imaging and MusicGen AI. <br>It's tailored for radio producers and music creators, offering new levels of creativity and efficiency by Bilsimaging. [Try our GPTs](https://chat.openai.com/g/g-65x53n87E-radio-imaging-musicgen-ai).", unsafe_allow_html=True)
st.markdown("If you appreciate my deployment and wish to support me, please consider a donation. Your support helps me continue providing value. Thank you for joining me on this journey! - Bilel Aroua")
st.markdown("For support ☕ [Buy me a Coffee](https://ko-fi.com/bilsimaging).")
st.image('https://storage.ko-fi.com/cdn/brandasset/kofi_button_dark.png', width=300, caption="Project Bilsimaigng")

# Hide Streamlit branding
st.markdown("<style>#MainMenu {visibility: hidden;} footer {visibility: hidden;}</style>", unsafe_allow_html=True)
