import speech_recognition as sr

def transcribe_wavefile(filename, language):
    recognizer = sr.Recognizer()

    # Load audio file
    with sr.AudioFile(filename) as source:
        audio = recognizer.record(source)

    try:
        # Use Google Web Speech API
        text = recognizer.recognize_google(audio, language=language)
    except sr.UnknownValueError:
        text = "Could not understand audio"
    except sr.RequestError:
        text = "API request failed"

    return text