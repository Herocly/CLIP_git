import torch
from clip import clip
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import Strawberry_dataset, create_dict  # 我自己定义的数据集
from Text_Attention import SelfAttentionFuser

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
            feats = feats / feats.norm(dim=-1,keepdim=True)
        return feats
        # 返回出来一个[n,768]的tensor

    def create_feature_dict(self, n):
        # 只对描述文本做clip编码，不加attention，可提前缓存
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
                disease1:[512维tensor]
                disease2:[512维tensor]
                ......
            }
            用来存储每个病害所对应的特征向量，以便于后来使用self-attention进行处理
        """
        return disease_features_dict
        # 最后将病的特征词典返回出来

    def attention_clip_train(self,image_folder,label_file,epochs=200,batch_size=30,lr=1e-5, save_path='Attention_Few_shot1.pth'):
        dataset = Strawberry_dataset(image_folder,label_file,self.preprocess)
        dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=True)

        loss_img = torch.nn.CrossEntropyLoss()
        loss_txt = torch.nn.CrossEntropyLoss()
        "loss(x, class) = -log(exp(x[class]) / sum(exp(x[i])))"

        # 只encode文本特征，不提前fuser【修改点1】
        disease_features_dict = self.create_feature_dict(2)
        "我们给每个病类的词典的feature目前提供3个语句"

        # 获取clip输出主维度，ViT-B/32一般是512
        embed_dim = next(iter(disease_features_dict.values())).shape[-1]
        attention_fuser = SelfAttentionFuser(input_dim=embed_dim, hidden_dim=256).to(self.device)

        # 优化器同时训练clip和attention_fuser【修改点2】
        optimizer = torch.optim.Adam(list(self.model.parameters()) + list(attention_fuser.parameters()), lr=lr)
        "此处为复制原来正常训练的代码，注释请见下文的正常训练"

        for epoch in range(epochs):
            for images, labels in tqdm(dataloader,desc=f"正在训练喵 {epoch+1}/{epochs}"):
                images = images.to(self.device)
                # labels = labels.to(self.device)
                "我们这边把labels存储成字符串的名字 labels = ['diseaseA','diseaseB'...]"

                image_features = self.model.encode_image(images)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                "这边对图像的处理方法和正常的都是一样，先编码再归一化"

                text_features = []
                # 【修改点3：fuser每batch前向一次，而非提前缓存】
                for label in labels:
                    # 从encode_dict拿到病的所有clip文本特征（N_text, embed_dim），attention_fuser处理，池化为一个
                    feats = disease_features_dict[label].to(self.device)
                    feats_fused, attn_weights = attention_fuser(feats)
                    # 在这里调用self_attention处理特征  feats_fused便为通过自注意力训练之后的向量
                    # attn_weights是权重参数[N N]维度的tensor
                    text_features.append(feats_fused.unsqueeze(0))
                text_features = torch.cat(text_features, dim=0)
                # torch.cat将存储的所有一维向量把他们叠加到一起，成为一个[batch_size,768]维的向量

                logit_scale = torch.nn.functional.softplus(self.model.logit_scale).clamp(min=1e-3, max=100)
                print("logit_scale:", logit_scale)
                # 还是每次都把参数打印出来

                logits_per_image = logit_scale * image_features @ text_features.t()
                logits_per_text = logits_per_image.t() # 转置即可得到文本与图像的矩阵

                lbl_range = torch.arange(len(images)).to(self.device)
                print("logits_per_image:", logits_per_image)
                print("logits_per_text:", logits_per_text)
                # 把每次的矩阵都打印一下，用来检测bug
                loss = ((loss_img(logits_per_image, lbl_range)) + loss_txt(logits_per_text, lbl_range)) / 2
                "loss_img 图像作为查询，文本是目标，loss_txt（文本为查询，图像是目标"
                print(f"loss = {loss}")

                optimizer.zero_grad()
                "这时我们会使用zero_grad使计算梯度每次计算后清零"
                loss.backward()
                'backward使梯度逐渐积累，不会自动清零'
                torch.nn.utils.clip_grad_norm_(list(self.model.parameters()) + list(attention_fuser.parameters()), max_norm=1.0)
                "梯度裁剪"
                optimizer.step()
                "optimizer使其自动训练，不断更新参数"
                # 训练+保存fuser
            torch.save(self.model.state_dict(), "clip_only.pth")
            print("save succeed!")

        # 保存clip参数




if __name__ == '__main__':
    study = ClipStudy(model_path="./ViT-B-32.pt")
    # study.normal_train(
    #     image_folder="dataset/images",
    #     label_file="dataset/output_range3.txt"
    # )
    study.attention_clip_train(
        image_folder="dataset/images",
        label_file="dataset/output_attention1.txt"
    )