import torch
from clip import clip
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import Strawberry_dataset, create_dict  # 我自己定义的数据集

class ClipStudy:
    def __init__(self, model_path, device=None, model_name="ViT-B-32"):
        # 设置计算设备
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # 模型选择['RN50', 'RN101', 'RN50x4', 'RN50x16', 'ViT-B/32', 'ViT-B/16']，对应不同权重
        self.model, self.preprocess = clip.load(model_path, device=self.device)  # 载入模型
        self.model_name = model_name

    def encode_descriptions(self, desc_list):
        # 将文本描述进行tokenize
        tokens = clip.tokenize(desc_list).to(self.device)
        with torch.no_grad():
            feats = self.model.encode_text(tokens)
            feats = feats / feats.norm(dim=-1)
        return feats
        # 返回出来一个[n,768]的tensor

    def create_feature_dict(self, n):
        disease_features_dict = {}
        disease_text_dict = create_dict(n)
        # 通过create_dict函数创建了一个字典
        """
            形式类似于
            {
                disease1:[feature1,feature2....]
                disease2:[feature1,feature2....]
            }
            desc_list就是一整行的列表，输入给encode变成向量
        """
        for disease, desc_list in disease_text_dict.items():
            feats = self.encode_descriptions(desc_list)
            disease_features_dict[disease] = feats
        """
            我们存储了一个词典disease_features_dict
            形式类似于
            {
                disease1:[768维tensor]
                disease2:[768维tensor]
                ......
            }
            用来存储每个病害所对应的特征向量，以便于后来使用self-attention进行处理
        """
        return disease_features_dict
        # 最后将病的特征词典返回出来



    def normal_train(self, image_folder, label_file, epochs=25, batch_size=30, lr=1e-5, save_path='Few_no_shot.pth'):
        dataset = Strawberry_dataset(image_folder, label_file, self.preprocess)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        "每次处理10对文本和图像的组合，shuffle表示训练前的同时把数据进行打乱"

        loss_img = torch.nn.CrossEntropyLoss()
        loss_txt = torch.nn.CrossEntropyLoss()
        "loss(x, class) = -log(exp(x[class]) / sum(exp(x[i])))"
        "非常标准的交叉熵损失函数，用于对比学习"
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        "设置一个优化器，gpt叫我写的，我也不知道有什么用2025/04/11"
        "现在我知道有什么用了，这是一个类型为adam的优化器，而右边的是学习率"
        "学习率会影响loss函数最终的计算 2025/04/15"

        for epoch in range(0, epochs):
            for image, text in tqdm(dataloader, desc=f"正在识别中 epoch {epoch+1}/{epochs}"):
                image = image.to(self.device)
                "将图像张量移动到指定的设备上"
                text = clip.tokenize(text[:50]).to(self.device)
                "自动截断成77个token"
                "后来我发现并不是严格按照单词去计算token，60个单词也有可能超过77个token"
                "这里暂且设置成前50个单词"

                image_features = self.model.encode_image(image)
                text_features = self.model.encode_text(text)

                eps = 1e-8
                image_features = image_features / (image_features.norm(dim=1, keepdim=True) + eps)
                text_features = text_features / (text_features.norm(dim=1, keepdim=True) + eps)
                # 进行一些简单的归一化

                logit_scale = torch.nn.functional.softplus(self.model.logit_scale).clamp(min=1e-3, max=100)
                print("logit_scale:", logit_scale)
                # 还是每次都把参数打印出来

                logits_per_image = logit_scale * image_features @ text_features.t()
                logits_per_text = logits_per_image.t()  # 转置即可得到文本与图像的矩阵
                #logits_per_text = logits_per_image.t()  # 转置即可得到文本矩阵

                labels = torch.arange(len(image)).to(self.device)
                print("logits_per_image:", logits_per_image)
                print("logits_per_text:", logits_per_text)
                # 把每次的矩阵都打印一下，用来检测bug
                loss = ((loss_img(logits_per_image, labels)) + loss_txt(logits_per_text, labels)) / 2
                "loss_img 图像作为查询，文本是目标，loss_txt（文本为查询，图像是目标"
                print(f"loss = {loss}")

                optimizer.zero_grad()
                "这时我们会使用zero_grad使计算梯度每次计算后清零"
                loss.backward()
                'backward使梯度逐渐积累，不会自动清零'
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                "梯度裁剪"
                optimizer.step()
                "optimizer使其自动训练，不断更新参数"

        torch.save(self.model.state_dict(), save_path)
        print(f"Model saved to {save_path}")



if __name__ == '__main__':
    study = ClipStudy(model_path="./ViT-B-32.pt")
    study.normal_train(
        image_folder="dataset/images",
        label_file="dataset/output_range3.txt"
    )