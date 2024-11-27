import Levenshtein as lev
from transformers import AutoTokenizer, T5ForConditionalGeneration, MarianMTModel, MarianTokenizer

# Grammar correction model
tokenizer = AutoTokenizer.from_pretrained("grammarly/coedit-large")
model = T5ForConditionalGeneration.from_pretrained("grammarly/coedit-large")

# Translation models (Hugging Face MarianMT for language translations)
TRANSLATORS = {
    "English": None,  # No translation needed
    "Gujarati": "Helsinki-NLP/opus-mt-gu-en",
    "Hindi": "Helsinki-NLP/opus-mt-hi-en",
    "Tamil": "Helsinki-NLP/opus-mt-ta-en",
    "French": "Helsinki-NLP/opus-mt-fr-en",
    "German": "Helsinki-NLP/opus-mt-de-en",
    "Japanese": "Helsinki-NLP/opus-mt-ja-en"
}

def load_translator(language):
    """Load the MarianMT model for the given language."""
    if language == "English":
        return None, None  # Return None for both model and tokenizer for English
    model_name = TRANSLATORS[language]
    if model_name:
        return MarianMTModel.from_pretrained(model_name), MarianTokenizer.from_pretrained(model_name)
    return None, None


def translate(text, model, tokenizer, direction="to_en"):
    """Translate text using MarianMT model."""
    if not model or not tokenizer:
        return text  # No translation required
    if direction == "to_en":
        # Translate to English
        input_ids = tokenizer(text, return_tensors="pt", padding=True).input_ids
        outputs = model.generate(input_ids)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    elif direction == "from_en":
        # Translate back to original language
        input_ids = tokenizer(text, return_tensors="pt", padding=True).input_ids
        outputs = model.generate(input_ids)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

def process_text(text, task_type):
    """
    Process the given text based on the task type using the Grammarly model.
    """
    task_prefixes = {
        "grammar_correction": "Fix grammatical errors in this sentence:",
        "coherence_correction": "Make this text coherent:",
        "rewrite_text": "Rewrite to make this easier to understand:",
    }
    input_text = f"{task_prefixes[task_type]} {text}"  # prepend the task prefix
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    outputs = model.generate(input_ids, max_length=1000)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def grammar_coherence_correction(text, language="English"):
    """
    Combine grammar correction, coherence correction, and text rewriting for multiple languages.
    
    Args:
        text (str): The text to process.
        language (str): The language of the text.

    Returns:
        dict: Contains the score, grammar corrected text, coherence corrected text, rewritten text.
    """
    # Load translator models based on language
    translator_model, translator_tokenizer = load_translator(language)

    # Step 1: Translate text to English if necessary
    text_in_english = translate(text, translator_model, translator_tokenizer, direction="to_en") if language != "English" else text

    # Step 2: Apply grammar correction and coherence checks in English
    grammar_corrected = process_text(text_in_english, "grammar_correction")
    coherence_corrected = process_text(grammar_corrected, "coherence_correction")
    rewritten = process_text(coherence_corrected, "rewrite_text")

    # Step 3: Translate the corrected text back to the original language
    corrected_text_in_original = translate(rewritten, translator_model, translator_tokenizer, direction="from_en") if language != "English" else rewritten

    # Calculate the similarity score
    score = calculate_overall_similarity_score(text, grammar_corrected, coherence_corrected, rewritten)
    
    return {
        "score": score,
        "grammar_corrected": grammar_corrected if language == "English" else translate(grammar_corrected, translator_model, translator_tokenizer, "from_en"),
        "coherence_corrected": coherence_corrected if language == "English" else translate(coherence_corrected, translator_model, translator_tokenizer, "from_en"),
        "rewritten": corrected_text_in_original,
        "original": text,
    }

def calculate_overall_similarity_score(original, grammar_corrected, coherence_corrected, rewritten):
    grammar_similarity = 1 - lev.distance(original, grammar_corrected) / max(len(original), len(grammar_corrected))
    coherence_similarity = 1 - lev.distance(original, coherence_corrected) / max(len(original), len(coherence_corrected))
    rewritten_similarity = 1 - lev.distance(original, rewritten) / max(len(original), len(rewritten))

    average_similarity = (grammar_similarity + coherence_similarity + rewritten_similarity) / 3
    return int(average_similarity * 100)
