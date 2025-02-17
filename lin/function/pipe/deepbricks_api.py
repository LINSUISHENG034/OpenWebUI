import requests

url = "https://api.deepbricks.ai/v1/chat/completions"
body = {
    "model": "claude-3.5-sonnet",
    "messages": [
        {
            "role": "user",
            "content": """Hello!"""
        }
    ],
    "stream": True
}
response = requests.post(url, headers={"Authorization": "Bearer sk-jRXDHkoPJ664Vz2iHnOREAL1pAeYCu6H1COOOkDFjj1914U0"}, json=body, stream=True)
for chunk in response.iter_lines():
    print(chunk)
