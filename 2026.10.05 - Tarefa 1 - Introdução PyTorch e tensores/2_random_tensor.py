import torch


if __name__ == "__main__":
    print("Initial Random Tensors:")
    random_tensor1 = torch.rand(3, 3)  # Uniformly distributed random tensor
    random_tensor2 = torch.randn(3, 3)  # Normally distributed random tensor

    print("T1:\n", random_tensor1)
    print("T2:\n", random_tensor2)

    print("\nMatrix Multiplication Results:")
    matmul_matrix_product_tensor = torch.matmul(random_tensor1, random_tensor2)
    mm_matrix_product_tensor = torch.mm(random_tensor1, random_tensor2)
    at_matrix_product_tensor = random_tensor1 @ random_tensor2

    print("\nMatmul Matrix Product:\n", matmul_matrix_product_tensor)
    print("\nMM Matrix Product:\n", mm_matrix_product_tensor)
    print("\n@ Matrix Product:\n", at_matrix_product_tensor)

    print("\nResults Comparison:")
    mm_matmul_equal = torch.allclose(
        mm_matrix_product_tensor, matmul_matrix_product_tensor)
    matmul_at_equal = torch.allclose(
        matmul_matrix_product_tensor, at_matrix_product_tensor)
    mm_at_equal = torch.allclose(
        mm_matrix_product_tensor, at_matrix_product_tensor)

    print("Matmul == MM:", mm_matmul_equal)
    print("Matmul == @:", matmul_at_equal)
    print("MM == @:", mm_at_equal)

    print("\nFinal Conclusion:")
    if mm_matmul_equal and matmul_at_equal and mm_at_equal:
        print("All matrix multiplication methods produce the same result.")
    else:
        print("There are differences between the matrix multiplication results.")
