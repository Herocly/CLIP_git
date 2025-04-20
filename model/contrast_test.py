import torch
from clip import clip
from dataset import Strawberry_dataset
from torch.utils.data import DataLoader, dataloader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_contrast():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 模型选择['RN50', 'RN101', 'RN50x4', 'RN50x16', 'ViT-B/32', 'ViT-B/16']，对应不同权重
    model, preprocess = clip.load("./ViT-B-32.pt", device=device)  # 载入模型

    weight_path = 'ViT-B-32-strawberry.pth'
    model.load_state_dict(torch.load(weight_path, map_location= device))
    # 加载自己的参数

    text_language={'Strawberry with Gray Mould disease,The strawberry fruit is covered by a dense, velvety gray mold, with soft, sunken spots and a mushy texture, often starting at wounds or bruises and spreading quickly, causing the fruit to appear shriveled and decayed.',
                    "Strawberry V-shaped brown leaf spot,The infected areas on the leaves expand in a V or U shape from the central vein toward the leaf base, with brown lesions that have dark brown edges, and the leaf shows signs of necrosis and wilting.",
                    "Strawberry with fertilizer damage disease,The leaf edges show small amounts of white fuzz, with black or yellow discoloration, and the leaves appear dull, with dry, brittle edges.",
                    "Strawberry blight,The strawberry roots turn yellow-brown, shrivel, and decay, while the heart leaves become yellow-green or yellow, curl tightly, shrink and narrow into a boat shape, and lose their glossy appearance.",
                    "Strawberry leaf spot caused by Ramularia grevilleana,Strawberry leaves develop small, round to irregular brown spots with a pale center, often surrounded by a yellowish halo, and as the disease progresses, the spots expand, leading to leaf tissue decay and premature leaf drop.",
                    "Strawberry calcium deficiency,Strawberry leaves show signs of curling and cupping, with younger leaves developing angular, necrotic spots along the margins, while older leaves may appear deformed and distorted with a dull, chlorotic yellowing between the veins.",
                    "Strawberry magnesium deficiency,Strawberry leaves exhibit interveinal chlorosis, where the tissue between veins turns yellow, while the veins remain green, and older leaves show signs of browning or necrosis at the tips and edges, often leading to premature leaf drop.",
                    "Strawberry Leaf Spot disease,Strawberry leaves show small, round to irregular brown spots with dark borders, often surrounded by a yellowish halo, and as the disease progresses, the spots enlarge, causing the leaf tissue to wither and die, leading to premature leaf drop.",
                    "Strawberry with anthracnose disease,Strawberry fruit shows sunken, water-soaked lesions that turn dark brown to black, often with concentric rings, while infected leaves develop irregular, dark spots with yellow halos, and the plant may exhibit early wilting and dieback in severe cases.",
                    "Normal starwberry,Healthy strawberry leaves are deep green, smooth, and uniform in shape, with vibrant, fresh foliage and no discoloration or spots, while the fruit is bright red, firm, and glossy with a well-formed shape and no signs of rot or blemishes.",
                    "Not related to strawberry"
                   }
    # 加载自己训练的类别文本

    text_token = clip.tokenize(text_language).to(device)
    # 并对其进行预处理，使其变成token形式



    test_dataset = Strawberry_dataset("dataset/few_shot/images", "dataset/few_shot/output.txt", preprocess)
    dataloader = DataLoader(test_dataset, batch_size=20, shuffle=True)
    "每次处理{batch_size =}对文本和图像的组合，shuffle表示训练前的同时把数据打乱"




    with torch.no_grad():
        for image,text in dataloader:
            # 先把dataloader里面的image和text的导入进去
            # images = torch.stack([preprocess(image) for image in images]).to(device)

            image = image.to(device)
            text = text.to(device)

            image_features = model.encode_image(image)
            text_features = model.encode_text(text_token)
            # 通过两个 encode ，获取图像和文本特征

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            # 进行归一化

            similarity = image_features @ text_features.T
            # 算出相似度矩阵[batch_size,num_text_language]

            prediction = similarity.argmax(dim = 1)
            # 取出图片对应最相似的文本索引
            """
            比如说算出来是一个[0.11,0.23,0.88,0.05]的矩阵/tensor
            prediction会返回一个数值为2的值
            但实际中不会一个向量一个向量去运算
            我们会直接得到n*len(text_language)的矩阵
            [0.11,0.23,0.88,0.05
            0.22,0.88,0.01,0.04
            ... ... ... ... ...]
            我们最后返回成一个列表 [2,1,...]用来保存标签的位置 
            """

            correction = 0
            correction = correction + (prediction == text).sum().item()
            """"
            prediction == text是在做布尔运算，猜对了就是1，猜错了就是0
            而text中存储的是正确的标签的位置
            若text中存储的是[3,1,....]
            若与上文的prediction运算就会变成[0,1....]
            item将标量值取出来
            将所有数值相加可以得到correction
            """
            total = 0
            total = total + len(text)

    Accuary = correction/total
    print("Accuracy:", Accuary)
    return Accuary

