# * Adicione suporte a batches para uma loss function da penúltima atividade (ex: MSE, etc...).

import torch


def binary_cross_entropy_loss(y_pred, y_train):
    return -torch.mean(y_train * torch.log(y_pred) + (1 - y_train) * torch.log(1 - y_pred))


if __name__ == "__main__":
    train_1 = torch.tensor([[0, 1], [1, 0]], dtype=torch.float32)
    preds_1 = torch.tensor([[0.1, 0.9], [0.8, 0.2]], dtype=torch.float32)

    train_2 = torch.tensor([[1, 1], [0, 0]], dtype=torch.float32)
    preds_2 = torch.tensor([[0.9, 0.8], [0.1, 0.2]], dtype=torch.float32)

    train_3 = torch.tensor([[1, 0], [0, 1]], dtype=torch.float32)
    preds_3 = torch.tensor([[0.7, 0.3], [0.2, 0.8]], dtype=torch.float32)

    batch_trains = torch.stack([train_1, train_2, train_3])
    batch_preds = torch.stack([preds_1, preds_2, preds_3])

    batch_loss = binary_cross_entropy_loss(batch_preds, batch_trains)
    print("Binary Cross Entropy Loss for the batch:", batch_loss)
