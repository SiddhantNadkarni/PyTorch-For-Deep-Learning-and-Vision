import torch
from torch.nn import Linear

torch.manual_seed(1) #sets a seed to generate random numbers

model = Linear(in_features = 1, out_features = 1)
print(model.bias, model.weight)


x = torch.tensor([[2.0], [3.3]]) #note the extra brackets

print(model(x))