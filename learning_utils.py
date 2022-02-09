import torch
import torch.nn.functional as F


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

def train_teacher(data, model, device, loss_fn, optimizer):
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

def train_student(data, student_model, teacher_model, device, loss_fn, optimizer):
    size = len(data.dataset)
    student_model.train()
    mean_loss = 0
    correct = 0
    current = 0
    for batch, (X, y) in enumerate(data):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = student_model(X)
        correct += (pred.argmax(dim=1) == y).sum().item()
        loss = calculate_kd_loss(pred, teacher_model(X), y, loss_fn)

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


def calculate_kd_loss(y_pred_student, y_pred_teacher, y_true, loss_fn):
        loss = 0.5 * loss_fn(y_pred_student, y_true)
        loss += 0.5 * loss_fn(
            y_pred_student, F.softmax(y_pred_teacher, dim=1))
        return loss