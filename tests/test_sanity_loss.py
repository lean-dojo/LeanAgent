# tests/test_sanity_loss.py
import torch
import torch.nn as nn

def test_dummy_loss_decreases():
    """
    Sanity-check that a mini training step actually lowers loss.
    Uses a 2-layer MLP on random data so it runs in <2 s on CPU.
    """
    torch.manual_seed(0)

    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 1)
    )

    x = torch.randn(32, 4)
    y = torch.randn(32, 1)

    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    # loss before training
    loss_start = criterion(model(x), y).item()

    # one quick training step
    optimizer.zero_grad()
    criterion(model(x), y).backward()
    optimizer.step()

    # loss after training
    loss_end = criterion(model(x), y).item()

    assert loss_end < loss_start, (
        f"loss did not decrease: start={loss_start:.4f}, end={loss_end:.4f}"
    )