import streamlit as st
from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI()

# Supported file formats
SUPPORTED_FORMATS = ["flac", "m4a", "mp3", "mp4", "mpeg", "mpga", "oga", "ogg", "wav", "webm"]

# Streamlit app
def main():
    st.title("Audio Transcription and Translation into English")
    st.write("Upload an audio file to transcribe and translate into English.")

    # File uploader
    uploaded_file = st.file_uploader("Choose an audio file", type=SUPPORTED_FORMATS)

    if uploaded_file is not None:
        st.write("Processing the file... keep patience, it may take few minutes.")

        try:
            # Save the uploaded file temporarily with the correct extension
            file_extension = uploaded_file.name.split('.')[-1]
            temp_file_name = f"temp_audio_file.{file_extension}"
            with open(temp_file_name, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Transcription
            with open(temp_file_name, "rb") as audio_file:
                transcription = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )

            # Translation
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a translator that translates sentences provided to you in any language to English. Also, you can analyze call transcriptions and separate conversations between a user and an AI call agent. You will mostly work with call conversations."},
                    {
                        "role": "user",
                        "content": transcription.text
                    }
                ]
            )

            # Display results
            translated_text = completion.choices[0].message.content
            st.subheader("Transcription and Translation Output:")
            st.text_area("Result", translated_text, height=200)

        except Exception as e:
            st.error(f"An error occurred: {e}")

        finally:
            # Clean up the temporary file
            if os.path.exists(temp_file_name):
                os.remove(temp_file_name)

if __name__ == "__main__":
    main()
