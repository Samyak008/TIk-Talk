from faster_whisper import WhisperModel

model_size = "large-v3"

model = WhisperModel(model_size, device="cpu", compute_type="int8")

# Language options for transcription, with their respective language codes.
LANGUAGE_MAP = {
    "English": "en",
    "Hindi": "hi",
    "Gujarati": "gu",
    "Tamil": "ta",
    "French": "fr",
    "German": "de",
    "Japanese": "ja"
}

def transcribe(language="en"):
    """
    Transcribe the audio file with support for multiple languages.
    
    Args:
        language (str, optional): The language of the audio file. Defaults to "en" for English.
                                  Supports 'en', 'hi', 'gu', 'ta', 'fr', 'de', 'ja'.
    
    Returns:
        str: The transcribed text.
    """
    # Ensure language is in the supported language map
    if language not in LANGUAGE_MAP.values():
        raise ValueError(f"Unsupported language: {language}. Please choose from {list(LANGUAGE_MAP.keys())}.")

    segments, _ = model.transcribe("tmp_file.wav", vad_filter=True, language=language)
    segments = list(segments)  # The transcription runs here
    return segments[0].text

# Example usage:
# To transcribe in Hindi, you would call: transcribe(language=LANGUAGE_MAP["Hindi"])
