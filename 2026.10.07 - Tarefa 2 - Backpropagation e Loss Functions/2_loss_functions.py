import torch
import torch.nn.functional as torch_loss
import pandas as pd

# Funções de perda (loss functions) para redes neurais

# ====================================================================================================
# Mean Squared Error (MSE) Loss is one of the most widely used loss functions for regression tasks. It calculates the average of the squared differences between the predicted values and the actual values. It is simple to understand and sensitive to outliers because the errors are squared which can affect the loss.


def mean_squared_error_loss(y_pred, y_train):
    return torch.mean((y_pred - y_train) ** 2)
# ====================================================================================================
# Mean Absolute Error (MAE) Loss is another commonly used loss function for regression. It calculates the average of the absolute differences between the predicted values and the actual values. It is less sensitive to outliers compared to MSE. But it is not differentiable at zero which can cause issues for some optimization algorithms.


def mean_absolute_error_loss(y_pred, y_train):
    return torch.mean(torch.abs(y_pred - y_train))
# ====================================================================================================
# Huber Loss combines the advantages of MSE and MAE. It is less sensitive to outliers than MSE and differentiable everywhere unlike MAE. It requires tuning of the parameter δ.


def huber_loss(y_pred, y_train, delta=1.0):
    error = y_pred - y_train
    is_small_error = torch.abs(error) <= delta
    squared_loss = 0.5 * error ** 2
    linear_loss = delta * (torch.abs(error) - 0.5 * delta)
    return torch.mean(torch.where(is_small_error, squared_loss, linear_loss))
# ====================================================================================================
# Binary Cross-Entropy Loss is also known as Log Loss and is used for binary classification problems. It measures the performance of a classification model whose output is a probability value between 0 and 1.


def binary_cross_entropy_loss(y_pred, y_train):
    return -torch.mean(y_train * torch.log(y_pred) + (1 - y_train) * torch.log(1 - y_pred))
# ====================================================================================================
# Categorical Cross-Entropy Loss is used for multiclass classification problems. It measures the performance of a classification model whose output is a probability distribution over multiple classes.


def categorical_cross_entropy_loss(y_logits, y_train):
    log_probs = torch.log_softmax(y_logits, dim=-1)
    return -torch.sum(y_train * log_probs, dim=-1)
# ====================================================================================================
# Dice Loss is used for image segmentation tasks and is particularly effective for imbalanced datasets. It measures the overlap between the predicted segmentation and the ground truth.


def dice_loss(y_pred, y_train, epsilon=1e-6):
    intersection = torch.sum(y_pred * y_train)
    union = torch.sum(y_pred) + torch.sum(y_train)
    return 1 - (2 * intersection + epsilon) / (union + epsilon)


# ====================================================================================================
if __name__ == "__main__":
    print("Loss Functions Examples:")
    # Example data
    y_pred = torch.tensor([0.2, 0.7, 0.1], dtype=torch.float32)
    y_target = torch.tensor([0, 1, 0], dtype=torch.float32)

    # Example MSE
    custom_mse = mean_squared_error_loss(y_pred, y_target)
    torch_mse = torch_loss.mse_loss(y_pred, y_target)
    comparison_mse = torch.isclose(
        torch.tensor(custom_mse), torch.tensor(torch_mse))
    print("\n=== Mean Squared Error Loss ===")
    print("Mean Squared Error Loss (Custom):", custom_mse.item())
    print("Mean Squared Error Loss (Torch):", torch_mse.item())
    print("MSE Comparison (Custom vs Torch):", comparison_mse.item())

    # Example MAE
    print("\n=== Mean Absolute Error Loss ===")
    custom_mae = mean_absolute_error_loss(y_pred, y_target)
    torch_mae = torch_loss.l1_loss(y_pred, y_target)
    comparison_mae = torch.isclose(custom_mae, torch_mae)
    print("Mean Absolute Error Loss (Custom):", custom_mae.item())
    print("Mean Absolute Error Loss (Torch):", torch_mae.item())
    print("MAE Comparison (Custom vs Torch):", comparison_mae.item())

    # Example Huber Loss
    print("\n=== Huber Loss ===")
    delta = y_target.mean().item()*0.2
    custom_huber = huber_loss(
        y_pred, y_target, delta=delta)
    torch_huber = torch_loss.huber_loss(
        y_pred, y_target, delta=delta)
    comparison_huber = torch.isclose(custom_huber, torch_huber)
    print("Huber Loss (Custom):", custom_huber.item())
    print("Huber Loss (Torch):", torch_huber.item())
    print("Huber Loss Comparison (Custom vs Torch):", comparison_huber.item())

    # Example Binary Cross Entropy
    print("\n=== Binary Cross Entropy Loss ===")
    custom_bce = binary_cross_entropy_loss(
        y_pred, y_target)
    torch_bce = torch_loss.binary_cross_entropy(
        y_pred, y_target)
    comparison_bce = torch.isclose(custom_bce, torch_bce)
    print("Binary Cross Entropy Loss (Custom):", custom_bce.item())
    print("Binary Cross Entropy Loss (Torch):", torch_bce.item())
    print("BCE Comparison (Custom vs Torch):", comparison_bce.item())

    # Example Categorical Cross Entropy
    print("\n=== Categorical Cross Entropy Loss ===")
    custom_cce = categorical_cross_entropy_loss(y_pred, y_target)
    torch_cce = torch_loss.cross_entropy(
        y_pred, y_target, reduction='mean')
    comparison_cce = torch.isclose(custom_cce, torch_cce)
    print("Categorical Cross Entropy Loss (Custom):", custom_cce.item())
    print("Categorical Cross Entropy Loss (Torch):", torch_cce.item())
    print("CCE Comparison (Custom vs Torch):", comparison_cce.item())

    # Example Dice Loss
    print("\n=== Dice Loss ===")
    custom_dice = dice_loss(y_pred, y_target)
    print("Dice Loss (Custom):", custom_dice.item())
