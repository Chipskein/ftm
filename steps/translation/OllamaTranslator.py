from ollama import Client, ResponseError
from .Translator import Translator
from profiler.ResourceMonitor import ResourceMonitor


LANG_NAMES: dict[str, str] = {
    "en": "English",
    "pt": "Portuguese",
    "ja": "Japanese"
}

MODEL = "translategemma:4b"
AVAILABLE_MODEL = [
    "translategemma:4b",
    "translategemma:12b"
]

class OllamaTranslator(Translator):
    """
    Translates text between any supported language pair using the
    translategemma model via Ollama.

    Requires Ollama to be running locally with translategemma pulled:
        ollama pull translategemma:4b
    """

    def __init__(
            self, 
            source_lang: str = "ja", 
            model: str = MODEL, 
            ollama_host: str = "http://localhost:11434",
            ollama_model_temperature: float = 0.0,
            monitor: ResourceMonitor | None = None
        ):
        """
        Args:
            source_lang : source language code (e.g. 'ja', 'en')
            model       : Ollama model name (default: translategemma:4b)
            ollama_host : Host URL for Ollama API (default: http://localhost:11434)
            ollama_model_temperature : Temperature for Ollama model (default: 0.0)
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
        self.monitor = monitor
        
        if self.monitor:
            self.monitor.set_label("Connecting with Ollama")

        self.source_lang = source_lang
        self.source_name = LANG_NAMES[source_lang]
        self.model = model
        self.ollama_host = ollama_host
        self.ollama_model_temperature = float(ollama_model_temperature)
        self._ollama_client = Client(
            host=self.ollama_host
        )
        
    def translate(self, text: str, lang: str, more_context: str = "") -> str:
        if self.monitor:
            self.monitor.set_label(f"Translating {text}")
            
        if not text.strip():
            return ""

        target_name = LANG_NAMES.get(lang.lower())
        if target_name is None:
            print(f"[OllamaTranslator] unknown language code: {lang}")
            return ""
        
        context_addition = ""
        if more_context:
            context_addition = (
                f"### Additional Context:\n"
                f"To help with nuance, keep the following context in mind:\n"
                f"{more_context}\n\n"
            )

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
            f"{context_addition}"
            f"Please translate the following {self.source_name} text into {target_name}:\n\n"
            f"{text}"
        )

        try:
            response = self._ollama_client.chat(
                model=self.model,
                stream=False,
                messages=[{"role": "user", "content": prompt}],
                options={ "temperature": self.ollama_model_temperature }
            )
            return response.message.content.strip()
        except ResponseError as e:
            print(f"[OllamaTranslator] Ollama error: {e.error}")
            return ""
        except Exception as e:
            print(f"[OllamaTranslator] unexpected error: {e}")
            return ""