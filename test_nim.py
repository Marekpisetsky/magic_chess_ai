import requests

BASE_URL = "http://127.0.0.1:8000"
MODEL = "nvidia/llama-3.1-nemotron-nano-vl-8b-v1"


def main():
    url = f"{BASE_URL}/v1/chat/completions"
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Dime en una frase que modelo eres."}
                ],
            }
        ],
        "temperature": 0.0,
        "max_tokens": 64,
    }

    print(f"Llamando a {url} ...")
    resp = requests.post(url, json=payload, timeout=60)
    print("Status:", resp.status_code)
    print("Respuesta cruda:\n", resp.text)

    resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    print("\n=== Mensaje del modelo ===")
    print(content)


if __name__ == "__main__":
    main()
