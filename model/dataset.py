import torch
from PIL import Image
from torch.utils.data import Dataset
import os
from BLM_API import gpt_descriptions
import re
from tqdm import tqdm


class Strawberry_dataset(Dataset):
    
    def __init__(self,image_dir,label_file,preprocess):
        """
        image_dir:图片的路径
        label_file:标签的文件
        preprocess:图片预处理的函数
        """

        self.image_dir = image_dir
        self.label_file = label_file
        self.preprocess = preprocess
        self.texts = []
        self.image_path = []
        "创建两个array，一个存储图片的文本，另外一个存储图片的地址"
        #with open(label_file,'r',encoding='utf-8') as f:
           #for line in f:
                #img_name,text = line.strip().split()
                #"strip是用来去掉一行的开头结尾空格换行符，而t是文字的标签"
                #self.image_path.append(os.path.join(self.image_dir,img_name))
                #self.texts.append(text)
        with open(label_file, 'r', encoding='gbk') as f:
            for line in f:
                parts = line.strip().split()
                #print(parts)
                if len(parts) < 2:
                    print(f"跳过格式错误的行: {line.strip()}")
                    continue
                img_name = parts[0]
                text = ' '.join(parts[1:])
                img_path = os.path.join(self.image_dir, img_name)


        # with open(label_file, 'r', encoding='utf-8') as f:
        #     for line in f:
        #         img_name, text = line.strip().split()
        #         self.image_path.append(os.path.join(image_dir, img_name))
        #         self.texts.append(text)

                if not os.path.exists(img_path):
                    print(f"警告: 找不到图片文件: {img_path}")
                    continue

                self.image_path.append(img_path)
                self.texts.append(text)
                #self.image_path.append(os.path.join(self.image_dir, img_name))
                #self.texts.append(text)
    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, idx):
        image = Image.open(self.image_path[idx]).convert("RGB")
        image = self.preprocess(image)
        text = self.texts[idx]
        return image, text



classify = {
        "10103": "Strawberry Gray Mould disease",
        "10115": "Strawberry V-shaped brown leaf spot disease",
        "10116": "Strawberry fertilizer damage disease",
        "10117": "Strawberry blight disease",
        "10118": "Strawberry leaf spot caused by Ramularia grevilleana disease",
        "10119": "Strawberry calcium deficiency disease",
        "10120": "Strawberry magnesium deficiency disease",
        "10121": "Strawberry Leaf Spot disease",
        "10122": "Strawberry anthracnose disease",
        "10123": "Normal strawberry without disease"
        }
def get_code(image_name):
    match = re.search(r'_(\d+)', image_name)
    # 通过re.search函数，找到下划线"_"后跟随的第一段较长的数字，从而把这个数字提取出来
    # 比如cut_img_10122_00000010.jpg 就会提取到10122
    if match:
        return match.group(1)
    else:
        return None

def get_features(disease:str):
    return gpt_descriptions(disease)

def create():
    folder_path = "D:\\cs_self\\1\\clip_git\\CLIP_git\\model\\dataset\\few_shot\\images"
    # 过段时间用os换一下相对路径，而不是用我电脑的绝对路径
    output_text = 'output.txt'
    file_list = os.listdir(folder_path)
    with open(output_text, 'w') as f:
        for(image_name) in tqdm(file_list,desc="生成文本ing喵"):
            # tqdm提供了一个进度条，用来给我提供进度
            disease_name = classify[get_code(image_name)]
            # 通过classify词组获得病名

            for i in range(2):
                prompt= get_features((disease_name))
                print(f"{image_name} {prompt}")
                f.write(f"{image_name} {prompt}\n")
            # 循环生成文本并写入txt，一个疾病对应三个文本

if __name__ == '__main__':
    # print(get_code("cut_img_10122_00000010.jpg"))
    create()
