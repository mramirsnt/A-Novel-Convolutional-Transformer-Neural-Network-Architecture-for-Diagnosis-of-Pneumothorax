import PIL
import matplotlib
import torch
import torch.nn as nn
from torch import Tensor
import torchvision
from torchvision import models, transforms, utils
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
from PIL import Image
import json
#matplotlib inline
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

from timm import create_model
import torch
import torchvision
import torchvision.transforms as T
from PIL import Image




class Interpret:
    def __init__(self, model, feature_extractor, image_path):
        super(Interpret)
        self.model = model
        self.feature_extractor = feature_extractor
        self.device = device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.img, self.img_tensor = self.load_image(image_path)
        self.model(self.img_tensor)

    def load_image(self, path):

        trans_img = transforms.PILToTensor()
        img = Image.open(path)
        image_size = (img.size[0], img.size[1])
        img = img.resize(image_size)
        t_img = trans_img(img)
        input_tensor = self.feature_extractor(t_img) # convert to tensor
        input_tensor = input_tensor.data
        input_tensor = Tensor(input_tensor['pixel_values'][0])
        input_tensor = input_tensor.unsqueeze(0)
        input_tensor = input_tensor.to(self.device)

        #print('img_tensor shape', img_tensor)
        return img, input_tensor

        #print('img_tensor shape', img_tensor)
        return img, img_tensor
    
    def show_patches(self, ):
        patches = self.model.vit.get_input_embeddings()(self.img_tensor)#patch_embed(self.img_tensor)  # patch embedding convolution
        print("Image tensor: ", self.img_tensor.shape)
        print("Patch embeddings: ", patches.shape)
        fig = plt.figure(figsize=(8, 8))
        fig.suptitle("Visualization of Patches", fontsize=24)
        #fig.add_axes()
        img = np.asarray(self.img)
        for i in range(0, 196):
            x = i % 14
            y = i // 14
            patch = img[y * 16:(y + 1) * 16, x * 16:(x + 1) * 16]
            ax = fig.add_subplot(14, 14, i + 1)
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            ax.imshow(patch)
    def visulize_position_embedding(self, ):
        patches = self.model.vit.get_input_embeddings()(self.img_tensor)#patch_embed(self.img_tensor)  # patch embedding convolution
        pos_embed = self.model.pos_embed
        print(pos_embed.shape)
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        fig = plt.figure(figsize=(8, 8))
        fig.suptitle("Visualization of position embedding similarities", fontsize=24)
        for i in range(1, pos_embed.shape[1]):
            sim = F.cosine_similarity(pos_embed[0, i:i + 1], pos_embed[0, 1:], dim=1)
            sim = sim.reshape((14, 14)).detach().cpu().numpy()
            ax = fig.add_subplot(14, 14, i)
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            ax.imshow(sim)

    def visualize_attention(self, ):
        patches = self.model.vit.get_input_embeddings()(self.img_tensor)#patch_embed(self.img_tensor)  # patch embedding convolution

        #patches = self.model.patch_embed(self.img_tensor)  # patch embedding convolution
        transformer_input = torch.cat((self.model.vit.embeddings.cls_token, patches), dim=1) + self.model.vit.embeddings.position_embeddings
        print("Transformer input: ", transformer_input.shape)
        print("Input tensor to Transformer (z0): ", transformer_input.shape)
        x = transformer_input.clone()
        for i, blk in enumerate(self.model.vit.encoder.layer):
            print("Entering the Transformer Encoder {}".format(i))
            print(x)
            x = blk(x)
            x = x[0]
            print('--output---')
        print('--------------------------------------------------------------------------')
        x = self.model.vit.layernorm(x)
        transformer_output = x[:, 0]
        print("Output vector from Transformer (z12-0):", transformer_output.shape)

        print("Transformer Multi-head Attention block:")
        attention = self.model.vit.encoder.layer[0].attention.attention
        print(attention)
        print("input of the transformer encoder:", transformer_input.shape)
        '''
        transformer_input_expanded = attention.qkv(transformer_input)[0]
        print("expanded to: ", transformer_input_expanded.shape)

        qkv = transformer_input_expanded.reshape(197, 3, 12, 64)  # (N=197, (qkv), H=12, D/H=64)
        print("split qkv : ", qkv.shape)
        q = qkv[:, 0].permute(1, 0, 2)  # (H=12, N=197, D/H=64)
        k = qkv[:, 1].permute(1, 0, 2)  # (H=12, N=197, D/H=64)
        kT = k.permute(0, 2, 1)  # (H=12, D/H=64, N=197)
        print("transposed ks: ", kT.shape)
        '''
        q = attention.query
        k = attention.key

        kT = attention.key.permute(0, 2, 1)
        attention_matrix = q @ kT
        print("attention matrix: ", attention_matrix.shape)
        plt.imshow(attention_matrix[3].detach().cpu().numpy())

        fig = plt.figure(figsize=(16, 8))
        fig.suptitle("Visualization of Attention", fontsize=24)
        fig.add_axes()
        img = np.asarray(self.img)
        ax = fig.add_subplot(2, 4, 1)
        ax.imshow(img)
        for i in range(7):  # visualize the 100th rows of attention matrices in the 0-7th heads
            attn_heatmap = attention_matrix[i, 100, 1:].reshape((14, 14)).detach().cpu().numpy()
            ax = fig.add_subplot(2, 4, i + 2)
            ax.imshow(attn_heatmap)
        print("Classification head: ", self.model.head)
        result = self.model.head(transformer_output)
        result_label_id = int(torch.argmax(result))
        plt.plot(result.detach().cpu().numpy()[0])
        plt.title("Classification result")
        plt.xlabel("class id")


    def interpret(self):
        self.show_patches()
        #self.visulize_position_embedding()
        #self.visualize_attention()
            