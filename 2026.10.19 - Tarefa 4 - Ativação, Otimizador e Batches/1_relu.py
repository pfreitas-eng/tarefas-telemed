# * Implemente uma função que recebe um tensor e seu output é a função de ativação ReLU com suporte a batches. Não utilize a função ReLU pronta!

import numpy as np
import torch


def relu(x):
    return torch.maximum(torch.tensor(0), x)


if __name__ == "__main__":
    x = torch.tensor([[-1, 2, -3], [4, -5, 6]])
    custom_relu_output = relu(x)
    torch_relu_output = torch.nn.functional.relu(x)
    comparacao = torch.all(custom_relu_output == torch_relu_output)

    print("Input:")
    print(x)

    print("Output:")
    print(custom_relu_output)

    print("Teste com PyTorch ReLU:")
    print(torch_relu_output)

    print("Comparação:")
    print("Custom = Torch:", comparacao)
