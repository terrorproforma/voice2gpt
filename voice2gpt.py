# voice2gpt
# records audio to transcribe & send to caht gpt. Uses OpenAI APIs.

import os
import time
import threading
import wave
import numpy as np
import sounddevice as sd
import keyboard
import openai
from config import API_KEY


AUDIO_DIRECTORY = "audio_recordings"
TRANSCRIPT_DIRECTORY = "audio_transcripts"
GPT_RESPONSE_DIRECTORY = "gpt_responses"
openai.api_key = API_KEY


def get_next_filename(directory, output_file):
    count = 1
    name, ext = os.path.splitext(output_file)
    new_output_file = os.path.join(directory, output_file)
    while os.path.exists(new_output_file):
        new_output_file = os.path.join(directory, f"{name}_{count}{ext}")
        count += 1
    return new_output_file


def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def save_audio(filename, recording, channels, rate):
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # For 16-bit audio
        wf.setframerate(rate)
        wf.writeframes(recording.tobytes())


def save_transcript(filename, transcript_text):
    with open(filename, "w") as f:
        f.write(transcript_text)


def save_gpt_response(filename, gpt_response):
    with open(filename, "w") as f:
        f.write(gpt_response)


def transcribe_audio_file(audio_file_path):
    with open(audio_file_path, "rb") as audio_file:
        transcript = openai.Audio.transcribe("whisper-1", audio_file)
    return transcript['text']


def generate_gpt_response(prompt, model="gpt-4"):
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful code writing assistant."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    return response['choices'][0]['message']['content'].strip()


def record_audio(output_file):

    # Parameters
    CHANNELS = 1
    RATE = 44100

    ensure_directory_exists(AUDIO_DIRECTORY)
    ensure_directory_exists(TRANSCRIPT_DIRECTORY)
    ensure_directory_exists(GPT_RESPONSE_DIRECTORY)

    print("Press 'space' to start recording, 'space' to stop recording, and 'q' to quit.")

    recording = None
    stop_recording = threading.Event()

    while True:
        if keyboard.is_pressed('space') and recording is None:
            print("Recording...")

            def callback(indata, frames, time, status):
                if not stop_recording.is_set():
                    recording.append(indata.copy())

            recording = []
            with sd.InputStream(samplerate=RATE, channels=CHANNELS, dtype='int16', callback=callback):
                while not keyboard.is_pressed('space'):
                    sd.sleep(100)

            print("Recording stopped. Press 's' to save the recording or 'r' to discard and record again.")
            time.sleep(0.5)  # Add a small delay to avoid multiple triggers

        elif keyboard.is_pressed('r') and recording is not None:
            recording = None
            stop_recording.clear()
            print("Discarded recording. Press 'space' to start recording again.")
            time.sleep(0.5)  # Add a small delay to avoid multiple triggers

        elif keyboard.is_pressed('s') and recording is not None:
            stop_recording.set()
            # Save the recorded data to a file
            next_filename = get_next_filename(AUDIO_DIRECTORY, output_file)
            recording = np.concatenate(recording, axis=0)
            save_audio(next_filename, recording, CHANNELS, RATE)
            print(f"Saved recording to {next_filename}")
            print("Press 'space' to start recording, 'space' to stop recording, and 'q' to quit.")
            recording = None
            stop_recording.clear()
            time.sleep(0.5)  # Add a small delay to avoid multiple triggers

            # transcribe the audio file
            transcript_text = transcribe_audio_file(next_filename)
            print(f"Transcript: {transcript_text}")

            # save transcript to file
            transcript_filename = get_next_filename(TRANSCRIPT_DIRECTORY, f"transcript_{os.path.basename(next_filename)}.txt")
            save_transcript(transcript_filename, transcript_text)

            # Get GPT-4 response
            gpt_response = generate_gpt_response(transcript_text)
            print(f"GPT-4 response: {gpt_response}")

            # save GPT-4 response to file
            gpt_response_filename = get_next_filename(GPT_RESPONSE_DIRECTORY, f"gpt_response_{os.path.basename(next_filename)}.txt")
            save_gpt_response(gpt_response_filename, gpt_response)

        elif keyboard.is_pressed('q'):
            print("Exiting...")
            break

if __name__ == "__main__":
    output_file = "output.wav"
    record_audio(output_file)
