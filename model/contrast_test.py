import torch
from clip import clip
from dataset import Strawberry_dataset
from torch.utils.data import DataLoader, dataloader
import numpy as np
from user_statistics.statistics import VCounter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_contrast():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 模型选择['RN50', 'RN101', 'RN50x4', 'RN50x16', 'ViT-B/32', 'ViT-B/16']，对应不同权重
    model, preprocess = clip.load("./ViT-B-32.pt", device=device)  # 载入模型

    #weight_path = 'Few_no_shot.pth'
    #model.load_state_dict(torch.load(weight_path, map_location= device))
    # 加载自己的参数
    model.eval()


    text_language = [
        "Strawberry Gray Mould disease",
        "Strawberry V-shaped brown leaf spot disease",
        "Strawberry fertilizer damage disease",
        "Strawberry blight disease",
        "Strawberry leaf spot caused by Ramularia grevilleana disease",
        "Strawberry calcium deficiency disease",
        "Strawberry magnesium deficiency disease",
        "Strawberry Leaf Spot disease",
        "Strawberry anthracnose disease",
        "Normal strawberry without disease"
    ]
    text_token = clip.tokenize(text_language).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_token)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    test_dataset = Strawberry_dataset("dataset/images", "dataset/test.txt", preprocess)

    correction = 0
    count = 0
    with torch.no_grad():
        counter = VCounter()
        text_features = model.encode_text(text_token)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        test_dataset = Strawberry_dataset("dataset/images", "dataset/test.txt", preprocess)
        for image, label in test_dataset:   # label为类别名称字符串
            image = image.to(device)
            if image.ndim == 3:
                image = image.unsqueeze(0)
            image_features = model.encode_image(image)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            logits = image_features @ text_features.t()   # [1, num_class]
            # 老样子还是使用image_features 和text_feature做矩阵相乘得到 logits

            probs = logits.softmax(dim=-1).cpu().numpy()[0]
            # 对所有类别做归一化，得到概率和为1的分布
            pred_id = np.argmax(probs)
            # 返回最大概率的下标
            #print(f"GT: {label}    Pred: {text_language[pred_id]}")
            counter.addPair(label, text_language[pred_id],probs[pred_id])
            if label.strip().lower() == text_language[pred_id].strip().lower():
                # 避免比较空格之类的瑕疵
                correction += 1
            count += 1
    # print("Total: %d; Correct: %d; Acc: %.2f%%" % (count, correction, correction/count*100))
    counter.print()
    return count, correction



if __name__ == '__main__':
    test_contrast()