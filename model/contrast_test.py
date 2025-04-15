import torch
from torch import device

from clip import clip
from clip.model import build_model
from dataset import Strawberry_dataset
from torch.utils.data import DataLoader, dataloader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_contrast():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 模型选择['RN50', 'RN101', 'RN50x4', 'RN50x16', 'ViT-B/32', 'ViT-B/16']，对应不同权重
    model, preprocess = clip.load("./ViT-B-32.pt", device=device)  # 载入模型

    weight_path = 'ViT-B-32-strawberry.pth'
    model.load_state_dict(torch.load(weight_path, map_location= device))
    #加载自己的参数


    test_dataset = Strawberry_dataset("dataset/images", "dataset/tests/labels/test.txt", preprocess)
    dataloader = DataLoader(test_dataset, batch_size=10, shuffle=True)
    "每次处理10对文本和图像的组合，shuffle表示训练前的同时把数据打乱"

    # loss_img = torch.nn.CrossEntropyLoss()
    # loss_txt = torch.nn.CrossEntropyLoss()
    # "loss(x, class) = -log(exp(x[class]) / sum(exp(x[i])))"
    # "非常标准的交叉熵损失函数，用于对比学习"
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    # "设置一个优化器，gpt叫我写的，我也不知道有什么用"


    with torch.no_grad():
        for images,text in dataloader:
            #先把dataloader里面的image和text的导入进去
            images = torch.stack([preprocess(image) for image in images]).to(device)
            text_token = clip.tokenize(text).to(device)
            # 对 images 和text 进行预处理


            image_features = model.encode_image(images)
            text_features = model.encode_text(text_token)
            #通过两个 encode ，获取图像和文本特征

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            #进行归一化

            similarity = image_features @ text_features.T
            #算出相似度矩阵

            prediction = similarity.argmax(dim = 1)
            #取出图片对应最相似的文本编码

            correction = correction + (prediction == text).sum().item()
            #prediction == text是在做布尔运算，猜对了就是1，猜错了就是0，将所有数值相加可以得到correction

            total = total + len(text)

    Accuary = correction/total
    print("Accuracy:", Accuary)
    return Accuary

