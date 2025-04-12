import numpy as np
import torch



if __name__ == '__main__':
    a = torch.tensor([[2, 3], [5,  7.0]])
    b = a.norm(dim=1, keepdim=True)
    print(b)







