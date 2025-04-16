import torch

from clip import clip
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import json
from flask import jsonify

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")


def class_demo1(path , text_language):
    # 测试分类的demo
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 模型选择['RN50', 'RN101', 'RN50x4', 'RN50x16', 'ViT-B/32', 'ViT-B/16']，对应不同权重
    model, preprocess = clip.load("./ViT-B-32.pt", device=device)  # 载入模型

    image = preprocess(Image.open(path)).unsqueeze(0).to(device)

    #text_language = ["a router", "a while box", "a while cat"] #从外部输入
    text = clip.tokenize(text_language).to(device)




    with torch.no_grad():
        logits_per_image, logits_per_text = model(image, text)  # 第一个值是图像，第二个是第一个的转置
        probs = logits_per_image.softmax(dim=-1).cpu().numpy() #此处为计算

        idx = np.argmax(probs, axis=1)
        for i in range(image.shape[0]):#拿到最大值的下标
            id = idx[i]
            print('image {}\tlabel\t{}:\t{}'.format(i, text_language[id],probs[i,id]))
            print('image {}:\t{}'.format(i, [v for v in zip(text_language,probs[i])]))

def class_demo2():
    # 测试分类的demo

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 模型选择['RN50', 'RN101', 'RN50x4', 'RN50x16', 'ViT-B/32', 'ViT-B/16']，对应不同权重
    model, preprocess = clip.load("./ViT-B-32.pt", device=device)  # 载入模型

    image = preprocess(Image.open("./killbug.png")).unsqueeze(0).to(device)
    print(image.shape)

    text_language = ["a cropland", "a black cat", "a pole",'there is a pole in the cropland']
    text = clip.tokenize(text_language).to(device)
    # print('text ==',text)
    # print('text.shape ==',text.shape) # text.shape == torch.Size([4, 77])


    with torch.no_grad():
        logits_per_image, logits_per_text = model(image, text)  # 第一个值是图像，第二个是第一个的转置
        print(logits_per_image.shape)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        #print(probs.shape)#(1, 4)
        idx = np.argmax(probs, axis=1)
        for i in range(image.shape[0]):
            id = idx[i]
            print('image {}\tlabel\t{}:\t{}'.format(i, text_language[id],probs[i,id]))
            print('image {}:\t{}'.format(i, [v for v in zip(text_language,probs[i])]))


def CLIP_demo():
    # 定义图像转换操作
    transform = transforms.Compose([
        # transforms.Grayscale(num_output_channels=1),  # 转换为单通道灰度图像,如果是三通道图
        transforms.ToTensor(),  # 将 PIL 图像转换为 [0, 1] 范围内的 Tensor
        # transforms.Resize((28, 28)),  # 调整图像大小
        # transforms.Normalize([0.1307], [0.3081])  # 归一化,这是mnist的标准化参数
    ])

    # 读取 PNG 图片
    #image_path = './killbug.png'  # 替换为你的图片路径
    image_path = './CLIP.png'  # 替换为你的图片路径
    image = Image.open(image_path)  # 确保图像有三个通道 (RGB)

    # 应用转换并转换为 Tensor
    tensor_image = transform(image)

    plt.imshow(tensor_image.permute(1, 2, 0))
    plt.show()


def class_demo_post(path , text_language):
    # 测试分类的demo
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 模型选择['RN50', 'RN101', 'RN50x4', 'RN50x16', 'ViT-B/32', 'ViT-B/16']，对应不同权重
    model, preprocess = clip.load("./ViT-B-32.pt", device=device)  # 载入模型

    image = preprocess(Image.open(path)).unsqueeze(0).to(device)

    #text_language = ["a router", "a while box", "a while cat"] #从外部输入
    text = clip.tokenize(text_language).to(device)




    with torch.no_grad():
        logits_per_image, logits_per_text = model(image, text)  # 第一个值是图像，第二个是第一个的转置
        probs = logits_per_image.softmax(dim=-1).cpu().numpy() #此处为计算

        idx = np.argmax(probs, axis=1)
        for i in range(image.shape[0]):#拿到最大值的下标
            id = idx[i]
            
            return {
                    "success": True,
                    "result": '{}    {}    others:{}'.format(text_language[id],probs[i,id],[v for v in zip(text_language,probs[i])])
                }
            
def class_demo_strawberry_post(path):
    text_language=['Strawberry with Gray Mould disease',
                    "Strawberry V-shaped brown leaf spot",
                    "Strawberry with fertilizer damage disease",
                    "Strawberry blight",
                    "Strawberry leaf spot caused by Ramularia grevilleana",
                    "Strawberry calcium deficiency",
                    "Strawberry magnesium deficiency",
                    "Strawberry Leaf Spot disease",
                    "Strawberry with anthracnose disease",
                    "Normal starwberry",
                    "Not related to strawberry"
                        ]
    chinese_text=["草莓灰霉病",
                    "草莓V型褐斑病",
                    "草莓肥害",
                    "草莓枯萎病",
                    "草莓拟盘多毛孢叶斑病",
                    "草莓缺钙",
                    "草莓缺镁",
                    "草莓蛇眼病",
                    "草莓炭疽病",
                    "正常草莓",
                    "与草莓无关"
    ]
    # 测试分类的dem
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 模型选择['RN50', 'RN101', 'RN50x4', 'RN50x16', 'ViT-B/32', 'ViT-B/16']，对应不同权重
    model, preprocess = clip.load("./ViT-B-32.pt", device=device)  # 载入模型

    weight_path = 'ViT-B-32-few.pth'
    model.load_state_dict(torch.load(weight_path, map_location= device))


    print(f"model = {model}")

    image = preprocess(Image.open(path)).unsqueeze(0).to(device)

    #text_language = ["a router", "a while box", "a while cat"] #从外部输入
    text = clip.tokenize(text_language).to(device)




    with torch.no_grad():
        logits_per_image, logits_per_text = model(image, text)  # 第一个值是图像，第二个是第一个的转置
        probs = logits_per_image.softmax(dim=-1).cpu().numpy() #此处为计算

        idx = np.argmax(probs, axis=1)
        for i in range(image.shape[0]):#拿到最大值的下标
            id = idx[i]
            


            temp_list = [(text,prob) for text, prob  in zip(chinese_text,probs[i])]
                #将文本与准确度打包生成list列表
            temp_list.sort(key=lambda x: x[1],reverse=True)
                #按照准确度从高到低排序


            #所有可能的结果总表
            dict_list = [{'index': '{}'.format(index),  #使用enumerate创建一个额外的序号作为主键
                          'text': text,                 #文本：对应的提示词
                          'prob': '{:.2f}'.format(prob*100)}    #准确度：重新格式化为带2位小数的百分比
                                for index,(text,prob) in enumerate(temp_list) 
                                    if prob > 0.00005]  #小于0.005%的结果忽略不计

            data =  {
                    "success": True,        #返回识别成功
                    "result": {             
                        "predict":'{}'.format(chinese_text[id]),        #最高准确度的结果
                        "prob":'{:.2f}'.format(probs[i,id]*100),        #对应的准确度
                        "all": dict_list                                #总表
                    }
                }
            
            return data


            # return {
            #         "success": True,
            #         "result": '{}    {}    others:{}'.format(chinese_text[id],probs[i,id],[v for v in zip(chinese_text,probs[i])])
            #     }


def zeroshot_strawberry_test(path):

    text_disease_name=["Healthy strawberry",
                       "Gray mold (Botrytis cinerea)",
                       "V-shaped brown leaf spot",
                       "Fertilizer damage",
                       "Blight",
                       "Ramularia leaf spot (caused by Ramularia grevilleana)",
                       "Calcium deficiency",
                       "Magnesium deficiency",
                       "General leaf Spot",
                       "Anthracnose",
                       "Powdery mildew",
                       "Lack of association with strawberries"]
    text_language=[]
    discribe_list_count=[]
    try:
        with open("./strawberry_disease.json",'r') as file:
            data_json = json.load(file)
            for disease_name in text_disease_name:
                temp_list_disease = data_json[disease_name]
                text_language.extend(temp_list_disease)
                #print(temp_list_disease)
                #print(len(temp_list_disease))
                discribe_list_count.append(len(temp_list_disease))
                #print(discribe_list_count)
    except:
        return({
            "success":False,
            "error":"Failed to load detect_list"
        })

    chinese_text=[  "健康的草莓",
                    "草莓灰霉病",
                    "草莓V型褐斑病",
                    "草莓肥害",
                    "草莓枯萎病",
                    "草莓拟盘多毛孢叶斑病",
                    "草莓缺钙",
                    "草莓缺镁",
                    "草莓蛇眼病",
                    "草莓炭疽病",
                    "草莓白粉病",
                    "图片与草莓无关"
    ]
    # 测试分类的dem
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 模型选择['RN50', 'RN101', 'RN50x4', 'RN50x16', 'ViT-B/32', 'ViT-B/16']，对应不同权重
    model, preprocess = clip.load("./ViT-L-14-336px.pt", device=device)  # 载入模型

    #weight_path = 'ViT-B-32-few.pth'
    #model.load_state_dict(torch.load(weight_path, map_location= device))


    #print(f"model = {model}")

    image = preprocess(Image.open(path)).unsqueeze(0).to(device)

    #text_language = ["a router", "a while box", "a while cat"] #从外部输入
    text = clip.tokenize(text_language).to(device)




    with torch.no_grad():
        logits_per_image, logits_per_text = model(image, text)  # 第一个值是图像，第二个是第一个的转置
        probs = logits_per_image.softmax(dim=-1).cpu().numpy() #此处为计算

        idx = np.argmax(probs, axis=1)




        for i in range(image.shape[0]):#拿到最大值的下标
            id = idx[i]
            

            prob_disease = [0 for _ in range(len(text_disease_name))]
            i_disease = 0
            i_sub = 0
            i_max_pos = 0
            for num_prob in discribe_list_count:
                for j in range(0,num_prob):
                    prob_disease[i_disease] += probs[i,i_sub]
                    if i_sub == id:
                        i_max_pos = i_disease
                    #print(f"j:{j}, i_disease:{i_disease}, i_sub:{i_sub}")
                    #print(i_max_pos, i_sub, i)
                    i_sub += 1
                #prob_disease[i_disease] /= num_prob
                i_disease += 1
            

            #sum_prob = 0
            #for tmp_prob in prob_disease:
                #sum_prob += tmp_prob

            #for i_prob in range(len(prob_disease)):
                #prob_disease[i_prob] = prob_disease[i_prob] / sum_prob

            temp_list = [(text,prob) for text, prob  in zip(chinese_text,prob_disease)]
            
                #将文本与准确度打包生成list列表
            temp_list.sort(key=lambda x: x[1],reverse=True)
                #按照准确度从高到低排序

            #print(temp_list)

            #所有可能的结果总表
            dict_list = [{'index': '{}'.format(index),  #使用enumerate创建一个额外的序号作为主键
                          'text': text,                 #文本：对应的提示词
                          'prob': '{:.2f}'.format(prob*100)}    #准确度：重新格式化为带2位小数的百分比
                                for index,(text,prob) in enumerate(temp_list)
                                    if prob > 0.00005 and index < 7]  #小于0.005%的结果忽略不计
            
            

            data =  {
                    "success": True,        #返回识别成功
                    "result": {             
                        "predict":'{}'.format(dict_list[0].get('text')),        #最高准确度的结果
                        "prob":'{}'.format(dict_list[0].get('prob')),        #对应的准确度
                        "all": dict_list                     #总表
                    }
                }
            #print(data)
            return data