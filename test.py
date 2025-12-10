from openai import OpenAI

# Cliente apuntando al servidor local de LM Studio
client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",  # cualquier string, LM Studio no valida la clave
)

# 1) Mostrar los modelos que ve LM Studio (para confirmar el ID)
models = client.models.list()
print("=== Modelos disponibles en LM Studio ===")
for m in models.data:
    print("-", m.id)

print("\n=== Probando chat.completions ===")

# OJO: usa exactamente el id que salga en la lista (por ejemplo "qwen2-vl-7b-instruct")
response = client.chat.completions.create(
    model="qwen2-v-7b-instruct",  # si falla, cambia este string por el ID exacto que te imprima arriba
    messages=[
        {"role": "system", "content": "Eres un asistente útil que responde en español."},
        {"role": "user", "content": "Dime en una sola frase quién eres."},
    ],
)

print("\n=== Respuesta del modelo ===")
print(response.choices[0].message.content)
