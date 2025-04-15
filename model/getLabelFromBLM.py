from openai import OpenAI
import os

os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key="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",
)


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

            Given the class name '{prompt}', generate as more unique short descriptions as you can, but don't generate more than 30.
            Each description should follow this format:
            a photo of '{prompt}' attribute

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
    print(gpt_labs("Strawberry with Gray Mould disease'"))