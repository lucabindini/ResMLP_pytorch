import torch


def train(data, model, device, loss_fn, optimizer):
    size = len(data.dataset)
    model.train()
    mean_loss = 0
    correct = 0
    current = 0
    for batch, (X, y) in enumerate(data):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        correct += (pred.argmax(dim=1) == y).sum().item()
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss = loss.item()
        print('loss: {:7f}  [{:{width}}/{:{width}}]'.format(
            loss, current, size, width=len(str(size))))
        current += len(X)

        mean_loss += loss
    return mean_loss/len(data), correct/size

    

def test(data, model, device, loss_fn):
    size = len(data.dataset)
    num_batches = len(data)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in data:
            X, y = X.to(device), y.to(device)
            pred = model(x)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print('Test Error:')
    print(f' Accuracy: {(100*correct):0.1f}%, Avg loss: {test_loss:8f}')
    return test_loss, correct