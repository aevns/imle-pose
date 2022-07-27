import numpy as np
import torch

probs = torch.tensor([.1,.01,.4,.05,.8])

sample = torch.distributions.categorical.Categorical(probs = probs)

base = torch.tensor([1/4,1/4,1/4,1/4])
base_sample = torch.distributions.categorical.Categorical(probs = base)

print(-torch.log(torch.sum(probs * probs / torch.sum(probs))))
print(torch.sum(probs * probs / torch.sum(probs)))

print(torch.sum(-torch.log(probs[probs>0]) * probs[probs>0] / torch.sum(probs[probs>0])))
print(torch.exp(-torch.sum(-torch.log(probs[probs>0]) * probs[probs>0] / torch.sum(probs[probs>0]))))

n = 1000000
freq = torch.zeros((5,))
for i in range(n):
    b = torch.randint(0,5,(1,))
    if (probs[b] < torch.rand(1)):
        continue
    else:
        freq[b] += 1

print(probs / torch.sum(probs))
print(freq / torch.sum(freq))