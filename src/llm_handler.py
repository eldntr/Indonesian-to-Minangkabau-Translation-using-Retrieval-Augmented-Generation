import os
import subprocess
import json
import config

def send_prompt_to_llm(prompt_final):
    """
    Mengirim prompt ke LLM menggunakan API OpenRouter.
    """
    api_key = os.getenv(config.OPENROUTER_API_KEY_ENV)
    if not api_key:
        print(f"API Key untuk OpenRouter tidak ditemukan. Pastikan variabel lingkungan '{config.OPENROUTER_API_KEY_ENV}' telah diatur.")
        return

    # Buat payload JSON dengan format yang valid
    payload = {
        "model": config.LLM_MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt_final
            }
        ]
    }

    curl_command = [
        "curl", "https://openrouter.ai/api/v1/chat/completions",
        "-H", "Content-Type: application/json",
        "-H", f"Authorization: Bearer {api_key}",
        "-d", json.dumps(payload)  # Gunakan json.dumps untuk memastikan JSON valid
    ]

    try:
        response = subprocess.check_output(curl_command, text=True)
        response_json = json.loads(response)  # Parse respons JSON
        answer = response_json.get("choices", [{}])[0].get("message", {}).get("content", "Tidak ada jawaban.")
        return answer
    except subprocess.CalledProcessError as e:
        print(f"Terjadi kesalahan saat mengirim permintaan ke OpenRouter: {e}")
    except json.JSONDecodeError:
        print("Gagal memproses respons JSON dari LLM.")
