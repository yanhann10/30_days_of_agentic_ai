import requests
import json
import os

response = requests.post(
  url="https://openrouter.ai/api/v1/chat/completions",
  headers={
    "Authorization": f"Bearer {os.environ.get('OLMO_API_KEY')}",
    "Content-Type": "application/json",
    "HTTP-Referer": os.environ.get('HTTP_REFERER', ''), # Optional. Site URL for rankings on openrouter.ai.
    "X-Title": os.environ.get('X_TITLE', ''), # Optional. Site title for rankings on openrouter.ai.
  },
  data=json.dumps({
    "model": "allenai/olmo-3-7b-instruct",
    "messages": [
      {
        "role": "user",
        "content": "What is the meaning of life?"
      }
    ]
  })
)