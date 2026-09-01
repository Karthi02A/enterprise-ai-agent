import requests
import json
import re

def clean_think_tags(text):
    if not text:
        return text
    # 1. Remove complete <think>...</think> blocks (case-insensitive, multi-line)
    text = re.sub(r'(?i)<think>.*?</think>', '', text, flags=re.DOTALL)
    # 2. If <think> is still in text (unclosed block), remove from <think> to the end
    if re.search(r'(?i)<think>', text):
        text = re.split(r'(?i)<think>', text)[0]
    # 3. If </think> is still in text (unclosed block), take everything after it
    if re.search(r'(?i)</think>', text):
        text = re.split(r'(?i)</think>', text)[-1]
    return text.strip()

def detect_language(text, api_key):
    """
    Detects if the input text language is English, Tamil, or Hindi.
    Returns 'en', 'ta', or 'hi'. Defaults to 'en' if unsure.
    """
    if not api_key or not api_key.strip():
        return "en"
    
    # Quick token check for obvious characters/keywords first if any
    text_lower = text.lower().strip()
    
    # Call Groq for precise short-text detection
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # We do a fast JSON classification prompt
    system_prompt = (
        "You are a language detection assistant. Your task is to detect the language of the provided text.\n"
        "Supported languages: 'en' (English), 'ta' (Tamil), 'hi' (Hindi).\n"
        "Return ONLY a JSON object with a single key 'lang' containing the language code ('en', 'ta', or 'hi').\n"
        "Do not write explanations, markdown block markers, or other text."
    )
    
    payload = {
        "model": "qwen/qwen3.6-27b",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Text: {text}"}
        ],
        "temperature": 0.0,
        "max_tokens": 30,
        "response_format": {"type": "json_object"}
    }
    
    import time
    max_retries = 3
    retry_delay = 1.0
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=8
            )
            if response.status_code == 200:
                res_json = response.json()
                content = res_json.get("choices", [{}])[0].get("message", {}).get("content", "")
                content = clean_think_tags(content)
                try:
                    data = json.loads(content)
                    lang = data.get("lang", "en").lower().strip()
                    if lang in ["en", "ta", "hi"]:
                        return lang
                except Exception:
                    if "ta" in content:
                        return "ta"
                    elif "hi" in content:
                        return "hi"
            elif response.status_code == 429:
                # Handle rate limit
                time.sleep(retry_delay)
                retry_delay *= 2
                continue
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= 2
                continue
    
    return "en"

def translate_text(text, target_lang, api_key):
    """
    Translates text to target_lang ('en', 'ta', 'hi') using Groq prompts.
    """
    if not api_key or not api_key.strip():
        return text
    
    if not text or not text.strip():
        return text
        
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    lang_names = {
        "en": "English",
        "ta": "Tamil",
        "hi": "Hindi"
    }
    
    target_name = lang_names.get(target_lang, "English")
    
    system_prompt = (
        f"You are a professional translator translating text to {target_name}.\n"
        "Instructions:\n"
        "1. Translate the user's text accurately to the target language.\n"
        "2. Preserve all formatting, code blocks, lists, headers, brackets, tables, and markdown structures.\n"
        "3. Output ONLY the translated text. Do not add intro, explanations, notes, or quotes."
    )
    
    payload = {
        "model": "qwen/qwen3.6-27b",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ],
        "temperature": 0.1,
        "max_tokens": 1500
    }
    
    import time
    max_retries = 3
    retry_delay = 1.0
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=25
            )
            if response.status_code == 200:
                res_json = response.json()
                translated = res_json.get("choices", [{}])[0].get("message", {}).get("content", "")
                translated = clean_think_tags(translated)
                if translated:
                    return translated
            elif response.status_code == 429:
                time.sleep(retry_delay)
                retry_delay *= 2
                continue
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= 2
                continue
            else:
                print(f"Translation error: {e}")
        
    return text
