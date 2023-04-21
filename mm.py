import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def myfunction():
  device = torch.device('cuda')
  print(device)
  data = torch.zeros((10000, 200, 300)).to(device)
  print(device)
  print(data.device)

def myfunc2():
  print('hi')

if __name__ == "__main__":
    myfunction()
    myfunc2()