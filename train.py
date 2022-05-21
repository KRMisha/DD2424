import torch
import config


def train(dataloader, model, loss_fn, optimizer, transform=None):
    model.train()

    total_loss = 0

    for X, y in dataloader:
        X, y = X.to(config.DEVICE), y.to(config.DEVICE)

        # Apply data transformations using GPU
        for i in range(len(X)):
            if transform is not None:
                rng_state = torch.get_rng_state()
                X[i] = transform(X[i])
                torch.set_rng_state(rng_state)
                y[i] = transform(y[i])

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # Return average loss
    return total_loss / len(dataloader)


def valid(dataloader, model, loss_fn):
    model.eval()

    total_loss = 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(config.DEVICE), y.to(config.DEVICE)
            pred = model(X)
            total_loss += loss_fn(pred, y).item()

    # Return average loss
    return total_loss / len(dataloader)
