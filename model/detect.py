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
            
def class_demo_strawberry_post(path , text_language):
    chinese_text=["营养不良的草莓",
                    "虫害草莓",
                    "缺钙型草莓",
                    "叶斑病草莓",
                    "白粉病草莓",
                    #"strawberry with other disease",
                    "正常的草莓果实",
                    "正常的草莓植株",
                    "与草莓无关",
    ]
    # 测试分类的dem
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 模型选择['RN50', 'RN101', 'RN50x4', 'RN50x16', 'ViT-B/32', 'ViT-B/16']，对应不同权重
    model, preprocess = clip.load("./ViT-B-32.pt", device=device)  # 载入模型

    weight_path = 'ViT-B-32-strawberry.pth'
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