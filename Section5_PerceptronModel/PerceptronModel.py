import torch
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

n_pts = 100
centers = [[-0.5, 0.5], [0.5, -0.5]]
X, y = datasets.make_blobs(n_samples = n_pts, random_state = 123, centers = centers, cluster_std = 0.4)
X_data = torch.Tensor(X)
y_data = torch.Tensor(y.reshape(100,1))
# print(X)
# print(y)

def scatter_plot():
	plt.scatter(X[y==0, 0], X[y==0, 1])
	plt.scatter(X[y==1, 0], X[y==1, 1])
	plt.show()


class Model(nn.Module):
	"""docstring for Model"""
	def __init__(self, input_size, output_size):
		super(Model, self).__init__()
		self.linear = nn.Linear(input_size, output_size)

	def forward(self, x):
		pred = torch.sigmoid(self.linear(x))
		return pred
	def predict(self, x):
		pred = self.forward(x)
		if pred >= 0.5:
			return 1
		if pred < 0.5:
			return 0

torch.manual_seed(2)
model = Model(2, 1)
# print(list(model.parameters()))
[w, b] = model.parameters()
w1, w2 = w.view(2)
b1 = b[0]

def get_params():
	return (w1.item(), w2.item(), b[0].item())

def plot_fit(title):
	plt.title = title
	w1, w2, b = get_params()
	x1 = np.array([-2.0, 2.0])
	x2 = (w1*x1 + b)/-w2
	plt.plot(x1, x2, 'r')
	scatter_plot()


plot_fit('Initial Model')


criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)
epochs = 2000
losses = []
for x in range(epochs):
	y_pred = model.forward(X_data)
	loss = criterion(y_pred, y_data)
	print("Epoch: ", x, "loss", loss.item())
	losses.append(loss.item())
	optimizer.zero_grad() #sets gradient to zero, since gradients accumulates after loss.backward call
	loss.backward() #computes gradient of loss function
	optimizer.step() #updates parameters

plt.plot(range(epochs), losses)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()
point1 = torch.Tensor([1.0, -1.0])
point2 = torch.Tensor([-1.0, 1.0])
plt.plot(point1.numpy()[0], point1.numpy()[1], 'ro')
plt.plot(point2.numpy()[0], point2.numpy()[1], 'ko')
print("Red Point positive probability = {}".format(model.forward(point1).item()))
print("Red Point positive probability = {}".format(model.predict(point1)))
print("Black Point positive probability = {}".format(model.forward(point2).item()))
print("Black Point positive probability = {}".format(model.predict(point2)))
plot_fit('Trained Model')
