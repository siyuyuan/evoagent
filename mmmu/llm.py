import os
import requests
import base64
from io import BytesIO
import pdb
import pathlib
import textwrap
import pdb
import google.generativeai as genai
import PIL.Image
import os

generation_config = {
    "temperature": 0,
}

safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE"
    }
]

openai_version = os.environ["OPENAI_API_VERSION"]
openai_base = os.environ["OPENAI_API_BASE"]
GPT4V_ENDPOINT = f"https://{openai_base}/openai/deployments/gpt-4-turbo-v/chat/completions?api-version={openai_version}"
GPT4V_KEY = os.environ["OPENAI_API_KEY"]
headers = {
    "Content-Type": "application/json",
    "api-key": GPT4V_KEY,
}


def llm_response(image, prompt, model):
    if model == 'gpt-4v':
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        img_data = buffer.getvalue()
        encoded_image = base64.b64encode(img_data).decode('utf-8')
        payload = {
            "model": "gpt-4-turbo-v",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encoded_image}"
                            }
                        }
                    ]
                }
            ],
            "temperature": 0,
            "max_tokens": 800
        }

        ind = 0
        while True:
            try:
                response = requests.post(GPT4V_ENDPOINT, headers=headers, json=payload)
                response.raise_for_status()
                break  # Will raise an HTTPError if the HTTP request returned an unsuccessful status code
            except requests.RequestException as e:
                ind = ind + 1
                if ind > 500:
                    return "Failed to response."
                continue
        return response.json()['choices'][0]['message']['content']
    elif model == 'gemini':
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        model = genai.GenerativeModel('gemini-pro-vision', generation_config=generation_config,
                                      safety_settings=safety_settings)
        ind = 0
        while True:
            try:
                # pdb.set_trace()
                response = model.generate_content([prompt, image])
                response_text = response.text
                return response_text
            except:
                print("Failed to make the request.")
                ind = ind + 1
                if ind > 10:
                    return "Failed to response."
                continue
