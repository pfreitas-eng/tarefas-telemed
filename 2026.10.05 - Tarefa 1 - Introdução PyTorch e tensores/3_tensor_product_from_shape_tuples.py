import torch


def tensor_product_from_shapes(shape1: tuple, shape2: tuple):
    tensor1 = torch.rand(shape1)
    tensor2 = torch.rand(shape2)

    print("\nTensor 1:\n", tensor1)
    print("\nTensor 2:\n", tensor2)

    at_product = tensor1 @ tensor2

    print("\nMatrix Multiplication (@):\n", at_product)

    return at_product


if __name__ == "__main__":
    print("=== Dot Product Example ===\n")
    shape1 = (3,)
    shape2 = (3,)
    tensor_product_from_shapes(shape1, shape2)

    print("\n\nIn case of two 1-dimensional tensors (vectors): The result is their dot product.")
    print("Prerequisite for dot product:")
    print("\tThe length of both vectors must match for the dot product.")
    print("Result of dot product:")
    print("\tThe result will be a scalar.")

    print("\n=== Matrix Multiplication Example ===\n")
    shape3 = (2, 4)
    shape4 = (4, 3)
    tensor_product_from_shapes(shape3, shape4)

    print("\n\nIn case of two 2-dimensional tensors (matrices): The result is their matrix multiplication.")
    print("Prerequisite for matrix multiplication:")
    print("\tThe number of columns in the first matrix must match the number of rows in the second matrix.")
    print("Result of matrix multiplication:")
    print("\tThe shape of the result will be (number of rows of the first matrix, number of columns of the second matrix).")

    print("\n=== Batched Matrix Multiplication Example ===\n")
    shape5 = (4, 2, 4)
    shape6 = (4, 4, 3)
    tensor_product_from_shapes(shape5, shape6)

    print("\n\nIn case of two 3-dimensional tensors: The result is their batched matrix multiplication.")
    print("Prerequisite for batched matrix multiplication:")
    print("\tThe number of columns in the last two dimensions of the first tensor must match the number of rows in the last two dimensions of the second tensor.")
    print("Result of batched matrix multiplication:")
    print("\tThe shape of the result will be (batch size, number of rows of the first matrix, number of columns of the second matrix).")
