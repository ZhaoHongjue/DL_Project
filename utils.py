import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import time

import torch
from torch import nn 
from torch.utils import data
from torch.nn import functional as F 

import torchvision
from torchvision import transforms

def plot(X, Y, labels = None, xlabel = None, ylabel = None, title = None, 
         grid = True, scatter = False, save = False):
    '''
    use to draw curves\\
    Y: tuple of array\\
    labels: tuple of label\\
    '''
    plt.clf()
    if scatter:
        draw_fn = plt.scatter
    else:
        draw_fn = plt.plot
        
    if type(Y) == tuple or type(Y) == list:
        for i in range(len(Y)):
            if labels is not None:
                draw_fn(X, Y[i], label = labels[i])
            else:
                draw_fn(X, Y[i])
    else:
        if labels is not None:
            draw_fn(X, Y, label = labels)
        else:
            draw_fn(X, Y)
    if labels is not None:
        plt.legend()
        
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if grid:
        plt.grid()
    if save:
        plt.savefig('./' + title + '.png')

def load_array(arrays, batch_size):
    dataset = data.TensorDataset(*arrays)
    return data.DataLoader(dataset, batch_size)

def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):  #@save
    '''draw picture in fashion mnist'''
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # 图片张量
            ax.imshow(img.numpy())
        else:
            # PIL图片
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes

def load_fashion_mnist(batch_size, resize = None):
    '''Download Fashion MNIST, and load it to memory'''
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root = './data',
        train = True,
        transform = trans,
        download = True
    )
    mnist_test = torchvision.datasets.FashionMNIST(
        root = './data',
        train = False,
        transform = trans,
        download = True
    )
    return (data.DataLoader(mnist_train, batch_size, shuffle=True),
            data.DataLoader(mnist_test, batch_size, shuffle=False))

def accuracy(y_hat, y):
    '''calculate the number of right predictions'''
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis = 1)
    cmp = y_hat.type(y.dtype) == y
    return cmp.sum()

def evaluate_accuracy(model, data_iter):
    if isinstance(model, nn.Module):
        model.eval()
    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(model(X), y), y.numel())    
    return metric[0] / metric[1]

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

'''------------------------------------------------------------------------------------------'''

def SGD(params, lr):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad
            param.grad.zero_()
            
def MSE(y_hat, y):
    return (0.5 * (y_hat - y.reshape(y_hat.shape))**2).mean()

def CrossEntropy(y_hat, y):
    return -torch.log(y_hat[range(len(y_hat)),y]).mean()

def softmax(o):
    return torch.exp(o)/torch.exp(o).sum(dim = 1).reshape(-1, 1)

class Timer: 
    """记录多次运行时间"""
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """启动计时器"""
        self.tik = time.time()

    def stop(self):
        """停止计时器并将时间记录在列表中"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """返回平均时间"""
        return sum(self.times) / len(self.times)

    def sum(self):
        """返回时间总和"""
        return sum(self.times)

    def cumsum(self):
        """返回累计时间"""
        return np.array(self.times).cumsum().tolist()

class Accumulator: 
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# if __name__ == '__main__':
#     a = np.arange(5)
#     b = 2 * a 
#     c = 3 * a
#     plot(a, (b, c), labels = ('b', 'c'), scatter=True)
#     plt.show()