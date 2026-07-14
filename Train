import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm

from models.resnet_raa import ResNet34


# -------------------- Hyperparameters --------------------
batch_size = 128
learning_rate = 0.1
num_epochs = 200
num_classes = 10

# -------------------- Model, loss, optimizer --------------------
cnn = ResNet34(num_classes=num_classes)
cnn = cnn.cuda()
criterion = nn.CrossEntropyLoss().cuda()

cnn_optimizer = torch.optim.SGD(cnn.parameters(), lr=learning_rate,
                                momentum=0.9, nesterov=True, weight_decay=5e-4)

scheduler = MultiStepLR(cnn_optimizer, milestones=[60, 120, 160], gamma=0.2)


# -------------------- Evaluation --------------------
def test(loader):
    cnn.eval()
    correct = 0.
    total = 0.
    for images, labels in loader:
        images = images.cuda()
        labels = labels.cuda()
        with torch.no_grad():
            pred = cnn(images)
        pred = torch.max(pred.data, 1)[1]
        total += labels.size(0)
        correct += (pred == labels).sum().item()
    cnn.train()
    return correct / total


# -------------------- Training loop --------------------
max_test_acc = 0.0

for epoch in range(num_epochs):
    xentropy_loss_avg = 0.
    correct = 0.
    total = 0.

    progress_bar = tqdm(train_loader)
    for i, (images, labels) in enumerate(progress_bar):
        progress_bar.set_description('Epoch ' + str(epoch))

        images = images.cuda()
        labels = labels.cuda()

        cnn.zero_grad()
        pred = cnn(images)
        loss = criterion(pred, labels)
        loss.backward()
        cnn_optimizer.step()

        xentropy_loss_avg += loss.item()
        pred = torch.max(pred.data, 1)[1]
        total += labels.size(0)
        correct += (pred == labels.data).sum().item()
        accuracy = correct / total

        progress_bar.set_postfix(
            xentropy='%.4f' % (xentropy_loss_avg / (i + 1)),
            acc='%.4f' % accuracy)

    test_acc = test(test_loader)
    if test_acc > max_test_acc:
        max_test_acc = test_acc

    tqdm.write('test_acc: %.4f (max: %.4f)' % (test_acc, max_test_acc))
    scheduler.step()
    print({
        'epoch': str(epoch),
        'train_acc': '%.4f' % accuracy,
        'test_acc': '%.4f' % test_acc,
        'xentropy': '%.4f' % (xentropy_loss_avg / (i + 1)),
        'max_test_acc': '%.4f' % max_test_acc
    })
