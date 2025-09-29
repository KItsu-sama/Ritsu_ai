import logging
import threading
try:
    import speech_recognition as sr
except Exception:
    sr = None

class SpeechListener(threading.Thread):                                                                           
    def __init__(self, on_text_callback=None):
        super().__init__(daemon=True)
        self.on_text = on_text_callback
        self.running = False
        self.recognizer = sr.Recognizer() if sr else None

    def run(self):
        if self.recognizer is None:
            logging.warning("SpeechRecognition not available. Mic disabled.")
            return
        try:
            with sr.Microphone() as source:
                logging.info("Speech listener: adjusting for ambient noise...")
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                self.running = True
                while self.running:
                    logging.debug("Listening for speech...")
                    audio = self.recognizer.listen(source, phrase_time_limit=6)
                    try:
                        text = self.recognizer.recognize_google(audio)
                        logging.info("Heard: %s", text)
                        if self.on_text:
                            self.on_text(text)
                    except Exception as e:
                        logging.debug("Speech recognition failed: %s", e)
        except Exception as e:
            logging.error("Microphone error: %s", e)

    def stop(self):
        self.running = False