from openai import OpenAI
import os
import json

# 设置代理（你已经设置好了）
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'

# 将API Key存入文件里进行打开
try:
    with open("./keys/openai_key.opk","r") as file:
        os.environ['OPENAI_API_KEY'] = file.readline()
except:
    os.environ['OPENAI_API_KEY'] = ""

#使用 OpenRouter 的 base_url
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

def gpt_descriptions(disease_name, n=1):
    prompt = f"""
You are a professional plant pathologist specializing in strawberries. 
Please write {n} different, precise, and visually distinctive English descriptions of strawberries showing the symptoms of {disease_name} for use in image-to-text retrieval. 
Here are the requirements:
1.Focus on describing what can be seen: color, shape, texture, location (leaf, fruit, stem), severity, and other visual clues. Each description should be 1-2 sentences and highlight slightly different aspects or appearances.
2. Avoid simply repeating the disease name in every sentence.
3.The symptoms are typically seen on the fruit/leaves/stems
4.Give me the sentence directly,do not add some "1." or else element
"""
    response = client.chat.completions.create(
        model="openai/gpt-4.1",    # 或"gpt-3.5-turbo"
        messages=[
            {"role": "system", "content": "You are an expert plant disease describer."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7  # 保证多样性
    )
    return response.choices[0].message.content






def gpt_labs_json(plant_name: str,feature_max_length: int, num_features_min: int, num_features_max: int): #plantname only support strawberry now.

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
        "content":f"You are a professional plant pathologist specializing in {plant_name}"
      },
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": f"""
You are helping build a visual recognition system using CLIP to identify {plant_name} diseases from images via zero-shot or few-shot learning.

Your task is to generate multiple English descriptive text prompts for each {plant_name} disease. These prompts will be compared with image embeddings by CLIP, so they must focus on distinct visual symptoms observable in real photographs.

Requirements for each description:
- Focus entirely on **visual characteristics**, such as color, texture, pattern, shape, affected parts of the plant, severity, and other visual clues.
- **Each sentence must include the word "{plant_name}"** to help CLIP ground the context.
- Avoid technical or biological explanations that are not visually evident, avoid simply repeating in every sentence.
- Avoid mentioning the disease name itself in every sentence; instead, describe what is visible.
- Highlight **slightly different aspects** or **appearances**. 
- Be concise, objective, and clear.
- Generate **{num_features_min} to {num_features_max} distinct sentences per disease**, each under {feature_max_length} words.

Output format:
Return a JSON object where each key is the disease name(or healthy strawberry) , and its value is a list of {num_features_min} to {num_features_max} descriptive sentences.

Diseases:
0. Healthy strawberry (normal condition, no disease)
1. Gray mold (Botrytis cinerea)  
2. V-shaped brown leaf spot
3. Fertilizer damage
4. Blight
5. Ramularia leaf spot (caused by Ramularia grevilleana)
6. Calcium deficiency
7. Magnesium deficiency
8. General leaf Spot
9. Anthracnose
10. Powdery mildew
  

Please generate multiple visual descriptions for each disease(or healthy strawberry), following the rules above.

            """
          }
        ]
      }
    ]
    )
    return result.choices[0].message.content




#作为主程序运行时
if __name__ == '__main__':
    # print(gpt_labs("Strawberry blight"))
    # print(gpt_descriptions("Strawberry blight"))
    # print("none.")
    #with open("./strawberry_disease.json","w") as json_file:
        #json_text = gpt_labs_json("strawberry",60,9,11)
        #print(json_text)
        #json_file.write(json_text)
    ask_gpt4("translate '草莓V型褐斑病' into english")
        