import os
import subprocess
import time
import shlex

import webbrowser
import json
import pyaudio
from vosk import Model, KaldiRecognizer


# Define the model path
model_path = "/Volumes/X/git/ece327/Archives/vosk-model-en-us-0.42-gigaspeech"
# model_path = "/Volumes/X/git/ece327/Archives/vosk-model-small-en-us-0.15"

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

is_speaking = False

def respond(response_text):
    global is_speaking
    is_speaking = True
    safe_response_text = shlex.quote(response_text)
    subprocess.call(['say', safe_response_text])
    is_speaking = False
    time.sleep(1)  # buffer time to let any speaker echo dissipate

def listen_for_command():
    global is_speaking
    print("Listening for commands...")

    # Wait until the system is done speaking before listening
    while is_speaking:
        time.sleep(0.1)

    # Process the audio stream in chunks
    while True:
        data = stream.read(4096, exception_on_overflow=False)
        if recognizer.AcceptWaveform(data):
            result = recognizer.Result()
            result_dict = json.loads(result)
            command = result_dict.get('text', '').lower()
            print("You said:", command)
            return command

def open_mac_app(app_name):
    try:
        subprocess.run(["open", "-a", app_name], check=True)
        print(f"{app_name} opened successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to open {app_name}: {e}")

def take_screenshot(filename="screenshot.png"):
    subprocess.run(["screencapture", filename])

tasks = []
listeningToTask = False

activationWords = ['Cool Buddy', 'Hey buddy', 'hey LUCY', 'lucy', 'TROY', 'hey TROY']

def main():
    global tasks
    global listeningToTask
    global is_speaking  # Declare the global variable

    while True:
        # Ensure the program is not speaking before listening for commands
        if not is_speaking:
            command = listen_for_command()

        # Check if the command contains any activation word
        if any(word.lower() in command for word in activationWords):
            activated = True
            while activated:
                respond("What can I do for you?")
                command = listen_for_command()

                if "add task" in command:
                    listeningToTask = True
                    respond("Sure, what is the task?")
                    command = listen_for_command()
                    tasks.append(command)
                    listeningToTask = False
                    respond(f"Task added: {command}. You now have {len(tasks)} tasks.")
                    activated = False  # Complete the action and exit the inner loop

                elif "list tasks" in command:
                    respond("Here are your tasks:")
                    for task in tasks:
                        respond(task)
                    activated = False  # Complete the action and exit the inner loop

                elif "take a screenshot" in command:
                    take_screenshot("screenshot.png")
                    respond("Screenshot taken and saved.")
                    activated = False  # Complete the action and exit the inner loop

                elif "open chrome" in command:
                    respond("Opening Google Chrome.")
                    webbrowser.open("http://www.google.com")
                    activated = False  # Complete the action and exit the inner loop

                elif "open main" in command:
                    respond("Opening FaceTime.")
                    open_mac_app("FaceTime")
                    activated = False  # Complete the action and exit the inner loop

                elif "open calendar" in command:
                    respond("Opening Calendar.")
                    open_mac_app("Calendar")
                    activated = False  # Complete the action and exit the inner loop

                elif "exit" in command:
                    respond("Exiting. Goodbye!")
                    return  # Exit the main loop and end the program

                elif listeningToTask:
                    tasks.append(command)
                    listeningToTask = False
                    respond(f"Task added. You have {len(tasks)} tasks.")
                    activated = False  # Complete the action and exit the inner loop

                else:
                    time.sleep(2)
                    respond("I'm here and listening.")
        else:
            respond("Say one of the activation words to start a command.")


if __name__ == "__main__":
    respond('All systems nominal.')
    respond('Welcome to Lovely University')
    try:
        main()
    except KeyboardInterrupt:
        print("\nGoodbye!")
