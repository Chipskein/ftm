from ollama import chat, ResponseError


LANG_NAMES: dict[str, str] = {
    "en": "English",
    "pt": "Portuguese",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "zh": "Chinese",
    "ko": "Korean",
    "ja": "Japanese",
    "it": "Italian",
    "ru": "Russian",
    "ar": "Arabic",
}

MODEL = "translategemma:4b"
AVAILABLE_MODEL = [
    "translategemma:4b",
    "translategemma:12b"
]

class OllamaTranslator:
    """
    Translates text between any supported language pair using the
    translategemma model via Ollama.

    Requires Ollama to be running locally with translategemma pulled:
        ollama pull translategemma:4b
    """

    def __init__(self, source_lang: str = "ja", model: str = MODEL):
        """
        Args:
            source_lang : source language code (e.g. 'ja', 'en')
            model       : Ollama model name (default: translategemma:4b)
        """
        if source_lang not in LANG_NAMES:
            raise ValueError(
                f"Unknown source language '{source_lang}'. "
                f"Supported: {list(LANG_NAMES.keys())}"
            )
        
        if model not in AVAILABLE_MODEL:
            raise ValueError(
                f"Unknown model '{model}'. "
                f"Supported: {AVAILABLE_MODEL}"
            )
        self.source_lang = source_lang
        self.source_name = LANG_NAMES[source_lang]
        self.model = model

    def translate(self, text: str, lang: str) -> str:
        """
        Translate text from source_lang to the target language.

        Args:
            text : source text to translate
            lang : target language code (e.g. 'en', 'pt')

        Returns:
            Translated string, or empty string on failure.
        """
        if not text.strip():
            return ""

        target_name = LANG_NAMES.get(lang.lower())
        if target_name is None:
            print(f"[OllamaTranslator] unknown language code: {lang}")
            return ""

        prompt = (
            f"You are a professional {self.source_name} ({self.source_lang}) "
            f"to {target_name} ({lang}) translator. "
            f"Your goal is to accurately convey the meaning and nuances of the "
            f"original {self.source_name} text while adhering to {target_name} "
            f"grammar, vocabulary, and cultural sensitivities. "
            f"The text is manga dialogue: preserve brevity, tone, and ambiguity. "
            f"Do not invent subjects or context not present in the source. "
            f"Produce only the {target_name} translation, "
            f"without any additional explanations or commentary.\n\n"
            f"Please translate the following {self.source_name} text into {target_name}:\n\n"
            f"{text}"
        )

        try:
            response = chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.message.content.strip()
        except ResponseError as e:
            print(f"[OllamaTranslator] Ollama error: {e.error}")
            return ""
        except Exception as e:
            print(f"[OllamaTranslator] unexpected error: {e}")
            return ""