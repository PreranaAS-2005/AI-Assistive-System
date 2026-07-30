import pyttsx3

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Convert text to speech
engine.say("Hello! Welcome to the AI Assistive System.")

# Speak the text
engine.runAndWait()