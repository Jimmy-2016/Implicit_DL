import matplotlib.pyplot as plt
import torch

from model import *


##
n_epochs = 10
batch_size_train = 128
batch_size_test = 16
log_interval = 1

lr = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

##
center = torch.tensor([.0, .0, .0])
model = MyNN(input_size=10, radius=1, center=center)  # example settings
criterion = torch.nn.MSELoss()  # Example loss function for a regression problem
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Example optimizer

# Dummy dataset
inputs = torch.randn(1000, 10)  # 100 samples, 10 features each
targets = torch.randn(1000, 1)  # 100 target points in 2D space
dataset = CustomDataset(inputs, targets)
data_loader = DataLoader(dataset, batch_size=batch_size_train, shuffle=True)

fig, axs = plt.subplots(2, 5, subplot_kw={'projection': '3d'})

train_loss = []
test_losses = []
ii = 0
for i in range(n_epochs):

   model.train()
   # print(i)
   for _, (tmpdata, tmptar) in enumerate(data_loader):
       # predict = model(tmpdata)

       optimizer.zero_grad()
       output = model(tmpdata, center)
       loss = criterion(output, tmptar)
       loss.backward()
       optimizer.step()


   train_loss.append(loss.item())

   if i % log_interval == 0:
       print('Train Epoch: {} \tLoss: {:.6f}'.format(
           i,  loss.item()))
       # plt.subplot(2, 5, ii)
       # ax = plt.axes(projection='3d')
       model.eval()
       with torch.no_grad():
           col = np.mod(ii, 5)
           row = 0 if ii <= 5 else 1

           x = model.linear(tmpdata)
           x = model.circle_projection(x, center)
           axs[row, col].scatter(x[:, 0].detach().numpy(), x[:, 1].detach().numpy(), x[:, 2].detach().numpy(), color='k', s=100)


       ii += 1



plt.show()
