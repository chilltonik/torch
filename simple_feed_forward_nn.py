import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# W2 = W2.cuda() or W2.to(device)
# W2 = W2.cpu() or W2.to("cpu")
# tensors must me on the same device


def activation(tensor: torch.FloatTensor) -> torch.FloatTensor:
    return (tensor >= 0.5).to(torch.float)


X: torch.tensor = torch.tensor([1, 0, 1], dtype=torch.float32, device=device)
W1: torch.tensor = torch.tensor(
    [[0.3, 0.3, 0], [0.4, -0.5, 1]], dtype=torch.float32, device=device
)
W2: torch.tensor = torch.tensor([-1, 1], dtype=torch.float32, device=device)


def iteration(
    X: torch.FloatTensor, W1: torch.FloatTensor, W2: torch.FloatTensor
) -> bool:
    Z: torch.FloatTensor = W1.matmul(X)
    print("Hidden neuron-layer sum:", Z)

    U: torch.FloatTensor = activation(Z)
    print("Hidden neuron-layer output:", U)

    Z2: torch.FloatTensor = torch.dot(W2, U)
    print("Hidden 2 neuron-layer output:", Z2)

    U2 = activation(Z2)
    return bool(U2)


if __name__ == "__main__":
    i: bool = iteration(X=X, W1=W1, W2=W2)
    print(i)
