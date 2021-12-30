import torch
import numpy as np

#tensor created directly from data

data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)

#tensor created from NumPy array

np_array = np.array(data)
x_np = torch.from_numpy(np_array)

x_ones = torch.ones_like(x_data) # retains the properties of x_data
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n")




#MOVING OPERATIONS FROM THE CPU TO THE GPU IF POSSIBLE
if torch.cuda.is_available():
    tensor = tensor.to('cuda')

#Standard NumPy like indexing and slicing

tensor = torch.ones(4, 4)
print('First row: ', tensor[0])
print('First column: ', tensor[:, 0])
print('Last column:', tensor[..., -1])
tensor[:,1] = 0
print(tensor)

#BRIDGE TENSOR TO NUMPY ARRAY

t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")
