import torch
from clip import clip
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from PIL import Image

from dataset import Strawberry_dataset
"我自己定义的数据集"


device = "cuda" if torch.cuda.is_available() else "cpu"
# 模型选择['RN50', 'RN101', 'RN50x4', 'RN50x16', 'ViT-B/32', 'ViT-B/16']，对应不同权重
model, preprocess = clip.load("./ViT-B-32.pt",device=device)  # 载入模型

dataset = Strawberry_dataset("dataset/few_shot/images","dataset/few_shot/few_shot.txt",preprocess)
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
"每次处理10对文本和图像的组合，shuffle表示训练前的同时把数据进行打乱"

loss_img = torch.nn.CrossEntropyLoss()
loss_txt = torch.nn.CrossEntropyLoss()
"loss(x, class) = -log(exp(x[class]) / sum(exp(x[i])))"
"非常标准的交叉熵损失函数，用于对比学习"
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
"设置一个优化器，gpt叫我写的，我也不知道有什么用2025/04/11"
"现在我知道有什么用了，这是一个类型为adam的优化器，而右边的是学习率"
"学习率会影响loss函数最终的计算 2025/04/15"

for epoch in range(0,15):
    for image,text in tqdm(dataloader):
        image = image.to(device)
        "将图像张量移动到指定的设备上"
        text = clip.tokenize(text).to(device)
        "自动截断成77个token"

        image_features = []
        text_features = []
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

        eps = 1e-8
        image_features = image_features / (image_features.norm(dim=1, keepdim=True) + eps)
        text_features = text_features / (text_features.norm(dim=1, keepdim=True) + eps)

        logit_scale = torch.nn.functional.softplus(model.logit_scale).clamp(min=1e-3, max=100)
        print("logit_scale:",logit_scale)
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()  # 转置即可得到文本与图像的相似度
        #logits_per_text = logits_per_image.t()  #转置即可得到文本矩阵

        labels = torch.arange(len(image)).to(device)
        print("logits_per_image:",logits_per_image)
        print("logits_per_text:",logits_per_text)
        loss = ((loss_img(logits_per_image,labels)) + loss_txt(logits_per_text,labels)) / 2
        "loss_img 图像作为查询，文本是目标，loss_txt（文本为查询，图象是目标"
        print(f"loss = {loss}")


        optimizer.zero_grad()
        "这时我们会使用zero_grad使计算梯度每次计算后清零"
        loss.backward()
        'backward使梯度逐渐积累，不会自动清零'
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        "梯度裁剪"
        optimizer.step()
        "optimizer使其自动训练，不断更新参数"

torch.save(model.state_dict(), 'ViT-B-32-strawberry.pth')