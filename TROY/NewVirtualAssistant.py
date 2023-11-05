import os
from gtts import gTTS
from playsound import playsound
import pyautogui
import webbrowser
import json
import pyaudio
from vosk import Model, KaldiRecognizer

# Redirect logs to a file
# log_file = "/path/to/vosk_log.txt"  # Replace with your desired log file path
# os.environ["VOSK_LOGLEVEL"] = "0"
#os.environ["VOSK_LOGFILE"] = log_file

# Define the model path
model_path = "/Volumes/X/git/ece327/Archives/vosk-model-en-us-0.42-gigaspeech"

# Check if the model directory exists
if not os.path.exists(model_path):
    print("Model path is incorrect or the model directory does not exist")
    exit(1)

# Load the model from the specified path
model = Model(model_path)

# Create a recognizer with the model
recognizer = KaldiRecognizer(model, 16000)

# Initialize pyaudio
p = pyaudio.PyAudio()

# Open the microphone stream
stream = p.open(format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                input=True,
                frames_per_buffer=8192)
stream.start_stream()

def listen_for_command():
    print("Listening for commands...")
    # Process the audio stream in chunks
    while True:
        data = stream.read(4096, exception_on_overflow=False)
        if recognizer.AcceptWaveform(data):
            result = recognizer.Result()
            result_dict = json.loads(result)
            command = result_dict.get('text', '').lower()
            print("You said:", command)
            return command

def respond(response_text):
    print(response_text)
    tts = gTTS(text=response_text, lang='en')
    tts.save("response.mp3")
    playsound("response.mp3")

tasks = []
listeningToTask = False

def main():
    global tasks
    global listeningToTask
    while True:
        command = listen_for_command()

        if command:
            if listeningToTask:
                tasks.append(command)
                listeningToTask = False
                respond("Adding " + command + " to your task list. You have " + str(len(tasks)) + " currently in your list.")
            elif "add a task" in command:
                listeningToTask = True
                respond("Sure, what is the task?")
            elif "list tasks" in command:
                respond("Sure. Your tasks are:")
                for task in tasks:
                    respond(task)
            elif "take a screenshot" in command:
                screenshot = pyautogui.screenshot()
                screenshot.save("screenshot.png")
                respond("I took a screenshot for you.")
            elif "open chrome" in command:
                respond("Opening Chrome.")
                #webbrowser.get(using='google-chrome').open("http://www.google.com")
                webbrowser.open("http://www.google.com")
            elif "exit" in command:
                respond("Goodbye!")
                break
            else:
                respond("Sorry, I'm not sure how to handle that command.")



if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nGoodbye!")
