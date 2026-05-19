from openai import OpenAI, AsyncOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

OPEN_ROUTER_CLIENT = OpenAI(
  base_url=os.getenv("OPENROUTER_BASE_URL"),
  api_key=os.getenv("OPENROUTER_API_KEY"),
)

ASYNC_OPENROUTER_CLIENT = AsyncOpenAI(
    base_url=os.getenv("OPENROUTER_BASE_URL"),
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

def test_openrouter_client():
    completion = OPEN_ROUTER_CLIENT.chat.completions.create(
        extra_headers={
            "HTTP-Referer": "<YOUR_SITE_URL>", # Optional. Site URL for rankings on openrouter.ai.
            "X-Title": "<YOUR_SITE_NAME>", # Optional. Site title for rankings on openrouter.ai.
        },
        extra_body={},
        model=os.getenv("OPENROUTER_MODEL_NAME"), # grok4
        messages=[
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": "What is in this image?"
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
                }
                }
            ]
            }
        ]
    )
    
    print(completion.choices[0].message.content)


if __name__ == "__main__":
    test_openrouter_client()
    print("Done")
    