import torch
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
import torch.nn.functional as F

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]) #Normalize all channels of image (mean) and (std)
training_dataset = datasets.MNIST(root = './data', train = True, transform = transform, download = True)

training_loader = torch.utils.data.DataLoader(dataset = training_dataset, batch_size = 100, shuffle = True)
def image_convert(tensor):
	image = tensor.clone().detach().numpy()
	image = image.transpose(1, 2, 0)
	print(image.shape)
	image = image*np.array((0.5, 0.5, 0.5)) + np.array((0.5, 0.5, 0.5))
	image = image.clip(0, 1) #to ensure pixel range is between 0 and 1
	return image 

dataiter =  iter(training_loader) #creates an object and allows us to go through training_loader one element at a time
images, labels = dataiter.next()
fig = plt.figure(figsize = (25, 4))

for x in np.arange(20):
	ax = fig.add_subplot(2, 10, x+1)
	plt.imshow(image_convert(images[x]))
	ax.set_title([labels[x].item()])
	plt.show()

class Classifier(nn.Module):
	"""docstring for Classifier"""
	def __init__(self, D_in, h1, h2, D_out):
		super().__init__() #provision of various methods and attributes
		self.linear1 = nn.Linear(D_in, h1) #3 layers of Nodes
		self.linear2 = nn.Linear(h1, h2)
		self.linear3 = nn.Linear(h2, D_out)
	def forward(self, x):
		x = F.relu(self.linear1(x))
		x = F.relu(self.linear2(x))
		x = self.linear3(x)
		return x

model = Classifier(784, 125, 65, 10)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)	
epochs = 12
running_losses_history = []
for x in range(epochs):
	for inputs, labels in training_loader:
		inputs = inputs.view()
