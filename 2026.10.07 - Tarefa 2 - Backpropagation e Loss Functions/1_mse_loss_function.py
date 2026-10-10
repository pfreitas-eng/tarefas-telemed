import torch


def mse_loss(y_pred, y_train):
    return torch.mean((y_pred - y_train) ** 2)


if __name__ == "__main__":
    y_pred = torch.rand(3)
    y_train = torch.rand(3)

    print(f'Predictions: {y_pred}')
    print(f'True values: {y_train}')

    loss = mse_loss(y_pred, y_train)
    print(f'Loss (MSE): {loss}')
