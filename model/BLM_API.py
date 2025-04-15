from openai import OpenAI
import os

# 设置代理（你已经设置好了）
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'

# ✅ 推荐：用环境变量设置 API Key
try:
    with open("./keys/openai_key.opk","r") as file:
        os.environ['OPENAI_API_KEY'] = file.readline()
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
            # "text": f"""
            # I am training a model using contrastive image-text learning.
            # Please act as an image description generator.
            #
            # Given the class name '{prompt}', generate 10 unique short descriptions.
            # Each description should follow this format:
            # "a photo of {prompt} attribute"
            #
            # Do not mention some artificial environments of this {prompt}
            #
            # The [attribute] should describe visual traits like color, shape, condition, or scene (e.g. 'ripe and red', 'covered with white mold', 'on a wooden table').
            #
            # No explanations, no numbering. Just return the 10 descriptions, one per line.
            # """
              "text":f"You are a professional plant pathologist specializing in strawberries. Please write 7 different, precise, and visually distinctive English descriptions of strawberries showing the symptoms of {prompt} for use in image-to-text retrieval. Focus on describing what can be seen: color, shape, texture, location (leaf, fruit, stem), severity, and other visual clues. Each description should be 1-2 sentences and highlight slightly different aspects or appearances. Avoid simply repeating the disease name in every sentence."
          }
        ]
      }
    ]
    )
    return result.choices[0].message.content

def get_descriptions(disease_name, n=7):
    prompt = f"""
You are a professional plant pathologist specializing in strawberries. Please write {n} different, precise, and visually distinctive English descriptions of strawberries showing the symptoms of {disease_name} for use in image-to-text retrieval. Focus on describing what can be seen: color, shape, texture, location (leaf, fruit, stem), severity, and other visual clues. Each description should be 1-2 sentences and highlight slightly different aspects or appearances. Avoid simply repeating the disease name in every sentence.
"""
    response = client.chat.completions.create(
        model="openai/gpt-4.1",    # 或"gpt-3.5-turbo"
        messages=[
            {"role": "system", "content": "You are an expert plant disease describer."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7  # 保证多样性
    )
    return response['choices'][0].message.conten


#作为主程序运行时
if __name__ == '__main__':
    print(gpt_labs("Strawberry blight"))
    # print(get_descriptions("Strawberry blight",7))