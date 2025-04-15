from openai import OpenAI
import os

# 设置代理（你已经设置好了）
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'

# ✅ 推荐：用环境变量设置 API Key
try:
    with open("./keys/openai_key.opk","r") as file:
        os.environ['OPENAI_API_KEY'] = file.read()
except:
    os.environ['OPENAI_API_KEY'] = ""

# ✅ 使用 OpenRouter 的 base_url
client = OpenAI(
    base_url="https://openrouter.ai/api/v1"
)

completion = client.chat.completions.create(
  extra_headers={
    "HTTP-Referer": "<YOUR_SITE_URL>", # Optional. Site URL for rankings on openrouter.ai.
    "X-Title": "<YOUR_SITE_NAME>", # Optional. Site title for rankings on openrouter.ai.
  },
  extra_body={},
  model="openai/gpt-4.1",
  messages=[
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "你是什么模型"
        }
        # {
        #   "type": "image_url",
        #   "image_url": {
        #     "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
        #   }
        # }
      ]
    }
  ]
)


def ask_gpt4(prompt: str):
    response = client.chat.completions.create(
    model="openai/gpt-4.1",
    messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def gpt_labs(prompt: str):

    result = client.chat.completions.create(
      extra_headers={
      "HTTP-Referer": "<YOUR_SITE_URL>",  # Optional. Site URL for rankings on openrouter.ai.
      "X-Title": "<YOUR_SITE_NAME>",  # Optional. Site title for rankings on openrouter.ai.
    },
    extra_body={},
    model="openai/gpt-4.1",
    messages=[
      {
        "role":"system",
        "content":"You are an agricultural expert."
      },
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": f"""
            I am training a model using contrastive image-text learning.
            Please act as an image description generator.

            Given the class name '{prompt}', generate 10 unique short descriptions.
            Each description should follow this format:
            "a photo of {prompt} attribute"
            
            Do not mention some artificial environments of this {prompt}

            The [attribute] should describe visual traits like color, shape, condition, or scene (e.g. 'ripe and red', 'covered with white mold', 'on a wooden table').

            No explanations, no numbering. Just return the 10 descriptions, one per line.
            """
          }
        ]
      }
    ]
    )
    return result.choices[0].message.content


#作为主程序运行时
if __name__ == '__main__':
    print(gpt_labs("Normal strawberry fruit"))
