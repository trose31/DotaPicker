"""This file converts the pytorch neural network to an onnx file, so
as it can be read by the Javascript in the website. """

import torch
from trainer import Net

def main():
  pytorch_model = Net()
  pytorch_model.load_state_dict(torch.load('pytorch_model.pt'))
  pytorch_model.eval()
  dummy_input = torch.zeros(1,250)
  torch.onnx.export(pytorch_model, dummy_input, 'onnx_model.onnx', verbose=True, opset_version = 10)


if __name__ == '__main__':
  main()
