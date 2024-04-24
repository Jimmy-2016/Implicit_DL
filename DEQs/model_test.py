import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from model import *
import torch.utils.data
import matplotlib.pyplot as plt
import torch.optim as optim






def disp_example(loader, indx):
    exmaple = enumerate(loader)
    batch_index, (data, target) = next(exmaple)

    print(data.shape)

    plt.imshow(data[indx, :, :, :].squeeze())

    print(target.shape)
    print(target[indx])


batch_size_train = 10
batch_size_test = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data/', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize((0.1307,), (0.3081,))
                               ])


    ),
    batch_size=batch_size_train, shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data/', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize((0.1307,), (0.3081,))
                               ])


    ),
    batch_size=batch_size_test, shuffle=True
)


exmaple = enumerate(test_loader)
batch_index, (data, target) = next(exmaple)

PATH = '../saved_model/model_iter.pth'
Conditional = 0

model = DEQModel().to(device)
model.load_state_dict(torch.load(PATH))



predict = model(data.view(data.shape[0], -1)).argmax(1)

# for i in range(6):
#     fig = plt.figure()
#     plt.imshow(data[i, :].reshape(28, 28))
#     plt.title(str(predict[i].item()))

for i in range(6):
    fig = plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(data[i, :].reshape(28, 28))
    plt.title(str(predict[i].item()))
    plt.subplot(1, 2, 2)
    plt.colorbar()
    x_input = model.flatten(data[i, :])
    x_deq = model.deq_layer(x_input)

    plt.imshow(x_deq.detach().numpy().reshape(28, 28))
    plt.colorbar()




plt.show()