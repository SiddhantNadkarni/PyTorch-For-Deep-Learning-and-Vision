import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

class LR(nn.Module):
	"""docstring for LR"""
	def __init__(self, input_size, output_size):
		super().__init__()
		self.linear = nn.Linear(input_size, output_size)
	def forward(self, x):
		pred = self.linear(x)
		return pred
		

torch.manual_seed(1)

model = LR(1, 1)
[w, b] = model.parameters()

X = torch.randn(100, 1)*10
y = X + 3*torch.randn(100, 1)
def getParams():
	return (w[0][0].item(),b[0].item())

def plot_fit(title):
	plt.title = title
	w1, b1 = getParams()
	x1 = np.array([-30, 30]) 
	y1 = w1*x1 + b1
	plt.plot(x1, y1, 'r')
	plt.scatter(X, y)
	plt.show()


def main():
	plot_fit('Intial model')










if __name__ == '__main__':
	main()