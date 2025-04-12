from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from clip import  clip
from clip import model

from PIL import Image

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x




class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        # ret : torch.Size([10, 50, 512])
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)  # n_head 头，d_model 表示维度。
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]  # 三个x表示Q K V计算值，x最后维度=n_head*d_model

    def forward(self, x: torch.Tensor):

        x = x + self.attention(self.ln_1(x))# torch.Size([10, 50, 512])

        x = x + self.mlp(self.ln_2(x))
        return x

    class CLIP(nn.Module):
        def __init__(self,
                     embed_dim: int,
                     # vision
                     image_resolution: int,
                     vision_layers: Union[Tuple[int, int, int, int], int],
                     vision_width: int,
                     vision_patch_size: int,
                     # text
                     context_length: int,
                     vocab_size: int,
                     transformer_width: int,
                     transformer_heads: int,
                     transformer_layers: int
                     ):
            super().__init__()

            self.context_length = context_length  # 77

            if isinstance(vision_layers, (tuple, list)):
                vision_heads = vision_width * 32 // 64
                self.visual = ModifiedResNet(
                    layers=vision_layers,
                    output_dim=embed_dim,
                    heads=vision_heads,
                    input_resolution=image_resolution,
                    width=vision_width
                )
            else:
                vision_heads = vision_width // 64
                self.visual = VisionTransformer(
                    input_resolution=image_resolution,
                    patch_size=vision_patch_size,
                    width=vision_width,
                    layers=vision_layers,
                    heads=vision_heads,
                    output_dim=embed_dim
                )

            self.transformer = Transformer(
                width=transformer_width,
                layers=transformer_layers,
                heads=transformer_heads,
                attn_mask=self.build_attention_mask()
            )

            self.vocab_size = vocab_size
            self.token_embedding = nn.Embedding(vocab_size, transformer_width)  #
            self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
            self.ln_final = LayerNorm(transformer_width)

            self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

            self.initialize_parameters()

        def initialize_parameters(self):
            nn.init.normal_(self.token_embedding.weight, std=0.02)
            nn.init.normal_(self.positional_embedding, std=0.01)

            if isinstance(self.visual, ModifiedResNet):
                if self.visual.attnpool is not None:
                    std = self.visual.attnpool.c_proj.in_features ** -0.5
                    nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                    nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                    nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                    nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

                for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                    for name, param in resnet_block.named_parameters():
                        if name.endswith("bn3.weight"):
                            nn.init.zeros_(param)

            proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
            attn_std = self.transformer.width ** -0.5
            fc_std = (2 * self.transformer.width) ** -0.5
            for block in self.transformer.resblocks:
                nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
                nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
                nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

            if self.text_projection is not None:
                nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

        def build_attention_mask(self):
            # lazily create causal attention mask, with full attention between the vision tokens
            # pytorch uses additive attention mask; fill with -inf
            mask = torch.empty(self.context_length, self.context_length)
            mask.fill_(float("-inf"))
            mask.triu_(1)  # zero out the lower diagonal
            return mask

        @property
        def dtype(self):
            return self.visual.conv1.weight.dtype

        def encode_image(self, image):
            return self.visual(image.type(self.dtype))

        def encode_text(self, text):
            # x 每个句子前面有值，有2个特殊符号[CLS]与[Seq]
            x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]，[3,77,512]
            x = x + self.positional_embedding.type(self.dtype)  # 位置编码直接赋可学习位置，添加位置信息[3,77,512]
            x = x.permute(1, 0, 2)  # NLD -> LND,[77,3,512]
            x = self.transformer(x)  # 共11个 和图像encode结构一致 [77,3,512]
            x = x.permute(1, 0, 2)  # LND -> NLD，[3,77,512]
            x = self.ln_final(x).type(self.dtype)
            # x.shape = [batch_size, n_ctx, transformer.width]
            # take features from the eot embedding (eot_token is the highest number in each sequence)
            # text.argmax(dim=-1) 句子最后有一个seq字段，是最大的，因此能获得句子个数数量
            # 对于很多预训练语言模型来说，它们会有一个特殊的开始标记 [CLS]，该标记的最终隐藏状态被用作整个句子的表示
            print(x[torch.arange(x.shape[0]), text.argmax(dim=-1)].shape)
            x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

            return x

        def forward(self, image, text):
            image_features = self.encode_image(image)
            text_features = self.encode_text(text)

            # normalized features,# 每一行sqr(a1^2+a2^2+...)
            image_features = image_features / image_features.norm(dim=1, keepdim=True)  # [batch_img,512]
            text_features = text_features / text_features.norm(dim=1, keepdim=True)  # [batch_text,512]

            # cosine similarity as logits
            logit_scale = self.logit_scale.exp()  # 可学习参数
            logits_per_image = logit_scale * image_features @ text_features.t()  # 特征相乘获得相似度
            logits_per_text = logits_per_image.t()  # 变成文本
            
            # shape = [global_batch_size, global_batch_size]
            return logits_per_image, logits_per_text


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
        # width相当于transform中的d_model
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        # x=[1,3,224,224]
        x = self.conv1(x)  # shape = [*, width, grid, grid] # 将图片分成[32,32]个patch [1,768,7,7]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2],合并高宽 [1,768,49]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width] ，更换位置 [1,49,768]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width],添加cls token[1,50,768]
        x = x + self.positional_embedding.to(x.dtype)  # 这里位置编码是可学习的参数，可能是切了path顺序让模型自己学习吧  [1,197,768]
        x = self.ln_pre(x)  # [1,197,768]

        x = x.permute(1, 0, 2)  # NLD -> LND  # [pixel,b,d_model]=[197,1,768]
        x = self.transformer(x)  # 多头transformer [197,1,768]
        x = x.permute(1, 0, 2)  # LND -> NLD  # [1,197,768]

        x = self.ln_post(x[:, 0, :])  # x[:, 0, :] 将所有信息汇聚到cls token中，只需前面来做下游任务 [1,768]

        if self.proj is not None:  # self.proj是可学习参数，维度为[768,512]
            x = x @ self.proj  # 通过学习参数将维度再次融合变成512特征，最终为[1,512]

        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int, #512
                 # vision
                 image_resolution: int, #224
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int, #768
                 vision_patch_size: int, #32
                 # text
                 context_length: int, #77
                 vocab_size: int, #49408
                 transformer_width: int, #512
                 transformer_heads: int, #8
                 transformer_layers: int #12
                 ):
        super().__init__()

        self.context_length = context_length  # 77
        #先定义一个context.length
        if isinstance(vision_layers, (tuple, list)):
            #首先判断vision layers是否属于一个元组,若满足isinstance则会使用resnet
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            #若if不通过我们采用的是vision transformer
            vision_heads = vision_width // 64
            ##多头注意力的头通过768 // 64 去算出来 所以头的维度就是64
            self.visual = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim
                #去进行一些数值的定义
            )

        self.transformer = Transformer(
            #定义transformer
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        #在embedding找到自己的512个向量，把文本变为三维向量
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        #设置位置编码
        self.ln_final = LayerNorm(transformer_width)
        #进行层归一化
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        ##初始化函数
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        #创建文本编码器的transformer的mask
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        #图像编码器
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        #文本编码器的定义
        # x 每个句子前面有值，有2个特殊符号[CLS]与[Seq]
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]，[3,77,512]
        x = x + self.positional_embedding.type(self.dtype)  # 位置编码直接赋可学习位置，添加位置信息[3,77,512]
        x = x.permute(1, 0, 2)  # NLD -> LND,[77,3,512]
        x = self.transformer(x)  # 共11个 和图像encode结构一致 [77,3,512]
        x = x.permute(1, 0, 2)  # LND -> NLD，[3,77,512]
        x = self.ln_final(x).type(self.dtype)
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # text.argmax(dim=-1) 句子最后有一个seq字段，是最大的，因此能获得句子个数数量
        # 对于很多预训练语言模型来说，它们会有一个特殊的开始标记 [CLS]，该标记的最终隐藏状态被用作整个句子的表示
        # print(x[torch.arange(x.shape[0]), text.argmax(dim=-1)].shape) #torch.Size([3, 512])
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)
        # print('image_features.shape ==', image_features.shape)# torch.Size([1, 512])
        # print('text_features.shape ==', text_features.shape) #torch.Size([3, 512])

        # normalized features,# 每一行sqr(a1^2+a2^2+...)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)  # [batch_img,512]
        text_features = text_features / text_features.norm(dim=1, keepdim=True)  # [batch_text,512]
        # print('image_features.shape ==', image_features.shape)# torch.Size([1, 512])
        # print('text_features.shape ==', text_features.shape) #torch.Size([3, 512])

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()  # 可学习参数
        logits_per_image = logit_scale * image_features @ text_features.t()  # 特征相乘获得相似度
        # .t(): 它会交换张量的第0维和第1维，即行和列的位置。对于更高维度的张量，.t() 只会对前两维进行转置，而不会影响其他维度
        logits_per_text = logits_per_image.t()  # 变成文本

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text



if __name__ == '__main__':
    # torch.float32
    # x = torch.randn(197, 1, 768)
    # model  = ResidualAttentionBlock(768, 12)
    # x = model(x) # 输出的结果形状依然是[10, 50, 512]
    # print(x.shape)

    # y = torch.randn(768)
    # x = torch.randn(1, 196, 768)
    # x = torch.cat(
    #     [y + torch.zeros(1, 1, 768), x],
    #     dim=1)  # shape = [*, grid ** 2 + 1, width],添加cls token[1,197,768]
    # print((y + torch.zeros(1, 1, 768)).shape) #torch.Size([1, 1, 768])
    # print(x.shape)# torch.Size([1, 197, 768])

    # mask = torch.empty(5, 5)
    # mask.fill_(float("-inf"))
    # mask.triu_(1)  # zero out the lower diagonal
    # print(mask)


    embed_dim = 512 #文本编码器的tansformer给予的嵌入维度
    image_resolution = 224 #图片的分辨率
    vision_layers = 12 #vision transformer有12层
    vision_width = 512 #图片的嵌入维度为512
    vision_patch_size = 32 #图片达成的patch的尺寸大小
    context_length = 77 #文本统一token的最大长度，如果没有77，剩余的部分用pad来补充
    #如果大于77会把多余的部分截取掉
    vocab_size = 49408 #"词表的大小"
    transformer_width = 512 #QKV的矩阵是512
    transformer_heads = 8 #分了8个头
    transformer_layers = 12

    #               512            224              12            768           32
    model = CLIP(embed_dim, image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers)  # 构建模型
    #       77            49408           512                  8                  12
    #通过这里定义了模型的初始化参数


    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("./ViT-B-32.pt", device=device)  # 载入模型

    # image = (1, 3, 224, 224)
    image = preprocess(Image.open("./CLIP.png")).unsqueeze(0).to(device)
    # text = (3, 77)
    text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)
    print(image.shape)
    print(text.shape)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]

    # model = CLIP(embed_dim, image_resolution, vision_layers, vision_width, vision_patch_size,
    #     context_length, vocab_size, transformer_width, transformer_heads, transformer_layers)
    #
    # # 输出模型各层及参数名
    # for name, module in model.named_modules():
    #     # if not name:  # 跳过顶层模型本身
    #     #     continue
    #     print(f"\nLayer: {name}")
    #     for param_name, param in module.named_parameters(recurse=False):
    #         print(f"  Parameter: {param_name}")

        # # 如果你想同时看到非参数属性（例如批归一化层的 running_mean 和 running_var）
        # for buffer_name, buffer in module.named_buffers(recurse=False):
        #     print(f"  Buffer: {buffer_name}")

