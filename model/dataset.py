import torch
from PIL import Image
from torch.utils.data import Dataset
import os



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
        with open(label_file, 'r', encoding='utf-8') as f:
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
