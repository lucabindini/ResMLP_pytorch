import torch


def train(data, model, device, loss_fn, optimizer):

    size = len(data.dataset)
    model.train()
    for batch, (X, y) in enumerate(data):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f'loss: {loss:>7f}  [{current:>5d}/{size:>5d}] images trained')

def test(data, model, device, loss_fn):

    size = len(data.dataset)
    num_batches = len(data)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in data:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f'Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n')