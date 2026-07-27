import torch

from improve.ns_qpeft import NSQPEFT

model = NSQPEFT()

dummy = torch.randn(8,12)

output = model(dummy)

print(output.shape)
print(output)