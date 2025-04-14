# import torch
# import
# from PIL import Image
# import numpy as np
# import matplotlib.pyplot as plt
# import torch
# import torch.nn.functional as F
# from torchvision import transforms
# from PIL import Image
#
#
# def class_demo1():
#     # 测试分类的demo
#
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     # 模型选择['RN50', 'RN101', 'RN50x4', 'RN50x16', 'ViT-B/32', 'ViT-B/16']，对应不同权重
#     model, preprocess = clip.load("../ViT-B-32.pt", device=device)  # 载入模型
#
#     image = preprocess(Image.open("../CLIP.png")).unsqueeze(0).to(device)
#
#     text_language = ["a diagram", "a dog", "a black cat"]
#     text = clip.tokenize(text_language).to(device)
#
#
#
#     with torch.no_grad():
#         logits_per_image, logits_per_text = model(image, text)  # 第一个值是图像，第二个是第一个的转置
#         probs = logits_per_image.softmax(dim=-1).cpu().numpy()
#
#         idx = np.argmax(probs, axis=1)
#         for i in range(image.shape[0]):
#             id = idx[i]
#             print('image {}\tlabel\t{}:\t{}'.format(i, text_language[id],probs[i,id]))
#             print('image {}:\t{}'.format(i, [v for v in zip(text_language,probs[i])]))
#
# def class_demo2():
#     # 测试分类的demo
#
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     # 模型选择['RN50', 'RN101', 'RN50x4', 'RN50x16', 'ViT-B/32', 'ViT-B/16']，对应不同权重
#     model, preprocess = clip.load("../ViT-B-32.pt", device=device)  # 载入模型
#
#     image = preprocess(Image.open("../killbug.png")).unsqueeze(0).to(device)
#
#     text_language = ["a cropland", "a black cat", "a pole",'a person']
#     text = clip.tokenize(text_language).to(device)
#
#
#
#     with torch.no_grad():
#         logits_per_image, logits_per_text = model(image, text)  # 第一个值是图像，第二个是第一个的转置
#         probs = logits_per_image.softmax(dim=-1).cpu().numpy()
#
#         idx = np.argmax(probs, axis=1)
#         for i in range(image.shape[0]):
#             id = idx[i]
#             print('image {}\tlabel\t{}:\t{}'.format(i, text_language[id],probs[i,id]))
#             print('image {}:\t{}'.format(i, [v for v in zip(text_language,probs[i])]))
#
#
# class CLIP_demo:
#     # 定义图像转换操作
#     transform = transforms.Compose([
#         # transforms.Grayscale(num_output_channels=1),  # 转换为单通道灰度图像,如果是三通道图
#         transforms.ToTensor(),  # 将 PIL 图像转换为 [0, 1] 范围内的 Tensor
#         # transforms.Resize((28, 28)),  # 调整图像大小
#         # transforms.Normalize([0.1307], [0.3081])  # 归一化,这是mnist的标准化参数
#     ])
#
#     # 读取 PNG 图片
#     image_path = '../CLIP.png'  # 替换为你的图片路径
#     image = Image.open(image_path)  # 确保图像有三个通道 (RGB)
#
#     # 应用转换并转换为 Tensor
#     tensor_image = transform(image)
#
#     plt.imshow(tensor_image.permute(1, 2, 0))
#     plt.show()
#
# if __name__ == '__main__':
#     # class_demo1()
#     # CLIP_demo()
#     class_demo2()