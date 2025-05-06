import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
import pandas as pd
import seaborn as sn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.sampler import SubsetRandomSampler
from dataset import HandDataset
from mynet import MyNet
from sklearn import metrics
from resnet18 import ResNet18

resnet = 0
RESNET = 0
CSV_NAME = 'HandInfo.csv'
DATASET_NAME = 'Hands'
VALIDATION = .2
TEST = .2
SHUFFLE = True
BATCH_SIZE = 8
RANDOM_SEED = 42
EPOCH = 11
LR = 0.001
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')     # Check cuda, if cuda gpu, if not cpu

val_loss_dict = {"x": [], "y": []}
train_loss_dict = {"x": [], "y": []}
val_acc_dict = {"x": [], "y": []}
train_acc_dict = {"x": [], "y": []}
train_times = []
val_times = []




# 2-Stream CNN Model
class TwoStreamCNN(nn.Module):
    def __init__(self, num_classes=8):
        super(TwoStreamCNN, self).__init__()
        
        # RGB stream
        self.rgb_stream = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten()
        )
        
        # Depth stream
        self.depth_stream = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten()
        )
        
        # Fusion layers
        self.fc = nn.Sequential(
            nn.Linear(128*32*32 + 128*32*32, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, rgb, depth):
        rgb_features = self.rgb_stream(rgb)
        depth_features = self.depth_stream(depth)
        combined = torch.cat((rgb_features, depth_features), dim=1)
        return F.log_softmax(self.fc(combined), dim=1)



def train_model(model, train_loader, validation_loader, test_loader):
    model = model.cuda()

    # OPTIMIZER AND SCHEDULER
    # optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9)
    # optimizer = optim.Adadelta(net.parameters(), lr=LR)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)

    val_acc, val_loss, _, _ = test(model, validation_loader)
    train_acc, train_loss, _, _ = test(model, train_loader)

    val_loss_dict['x'].append(0), val_loss_dict['y'].append(val_loss)
    train_loss_dict['x'].append(0), train_loss_dict['y'].append(train_loss)

    val_acc_dict['x'].append(0), val_acc_dict['y'].append(val_acc)
    train_acc_dict['x'].append(0), train_acc_dict['y'].append(train_acc)

    # TRAIN AND VALIDATE
    for epoch in range(1, EPOCH):
        train_start = time.time()
        train(model, train_loader, optimizer, epoch)
        train_end = time.time()
        train_times.append(train_end - train_start)

        val_start = time.time()
        val_acc, val_loss, _, _ = test(model, validation_loader)
        val_end = time.time()
        val_times.append(val_end - val_start)

        train_acc, train_loss, _, _ = test(model, train_loader)

        val_loss_dict['x'].append(epoch), val_loss_dict['y'].append(val_loss)
        train_loss_dict['x'].append(epoch), train_loss_dict['y'].append(train_loss)

        val_acc_dict['x'].append(epoch), val_acc_dict['y'].append(val_acc)
        train_acc_dict['x'].append(epoch), train_acc_dict['y'].append(train_acc)

        scheduler.step()

    test_acc, test_loss, ground_truths, test_results = test(model, test_loader)
    print(test_acc)
    get_confusion_matrix(ground_truths, test_results)
    plot_and_write(test_acc)
    print('Finished Training')
    torch.save(model.state_dict(), 'hand_model.pth')


def get_confusion_matrix(ground_truth, test_result):

    score = metrics.accuracy_score(ground_truth, test_result)
    # cls_report = metrics.classification_report(ground_truth, test_result)
    conf_mat = metrics.confusion_matrix(ground_truth, test_result)

    print('Accuracy: {:.3f}'.format(score))
    # print(cls_report)
    print(conf_mat)

    df_cm = pd.DataFrame(conf_mat, range(8), range(8))
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True, fmt='d', cmap=plt.get_cmap('jet'))  # font size
    plt.title("Confusion Matrix")
    plt.show()


def plot_and_write(test_acc):
    # plt.title("dropout1=0.25 dropout2=0.75")
    plt.plot(val_loss_dict['x'], val_loss_dict['y'], label="Validation")
    plt.plot(train_loss_dict['x'], train_loss_dict['y'], label="Train")
    plt.xticks(np.arange(0, 11, 1))
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # plt.title("dropout1=0.25 dropout2=0.75")
    plt.plot(val_acc_dict['x'], val_acc_dict['y'], label="Validation")
    plt.plot(train_acc_dict['x'], train_acc_dict['y'], label="Train")
    plt.xticks(np.arange(0, 11, 1))
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


def build_train_valid_test_subsets():

    # BUILD DATASET
    dataset = HandDataset(DATASET_NAME, CSV_NAME)
    print('######### Dataset class created #########')
    print('Number of images: ', len(dataset))

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    validation_split = int(np.floor(VALIDATION * dataset_size))
    test_split = int(np.floor(TEST * dataset_size))

    if SHUFFLE:
        np.random.seed(RANDOM_SEED)
        np.random.shuffle(indices)

    train_indices = indices[(validation_split + test_split):]
    val_indices = indices[validation_split:(validation_split + test_split)]
    test_indices = indices[:validation_split]

    return data_loader(dataset, train_indices), data_loader(dataset, val_indices), data_loader(dataset, test_indices)


def data_loader(dataset, indices):
    # Creating PT data samplers and loaders:
    sampler = SubsetRandomSampler(indices)
    return torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=6)


def train(model, train_loader, optimizer, epoch, running_loss=0.0, examples=0):
    model.train()
    for i, data in enumerate(train_loader, 0):
        # Get inputs and classes
        inputs, classes = data['image'].to(DEVICE), data['class'].to(DEVICE)

        # Zero parameter gradients
        optimizer.zero_grad()

        # forward, backward, optimize
        outputs = model(inputs)
        # loss = loss_criterion(outputs, classes)
        loss = F.nll_loss(outputs, classes)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        # Print statistics
        examples += BATCH_SIZE
        print('[%d, %5d] loss: %.7f' % (epoch + 1, i + 1, running_loss / examples))


def test(model, validation_loader, test_loss=0, correct=0):
    ground_truths = []
    test_results = []
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(validation_loader, 0):
            # Get inputs and classes
            inputs, classes = data['image'].to(DEVICE), data['class'].to(DEVICE)

            for i in classes.tolist():
                ground_truths.append(i)

            outputs = model(inputs)
            test_loss += F.nll_loss(outputs, classes).item()  # sum up batch loss
            # test_loss += F.binary_cross_entropy(outputs, classes, reduction='sum').item()  # sum up batch loss
            # test_loss += loss_criterion(outputs, classes).item()   # sum up batch loss
            pred = outputs.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(classes.view_as(pred)).sum().item()
            # correct += pred.eq(classes.argmax(dim=1, keepdim=True)).sum().item()

            for i in pred.tolist():
                test_results.append(i[0])

    test_loss /= len(validation_loader.sampler.indices)
    test_acc = 100. * correct / len(validation_loader.sampler.indices)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(validation_loader.sampler.indices),
        test_acc))

    return test_acc, test_loss, ground_truths, test_results


if __name__ == '__main__':
    # BUILD TRAIN-VALIDATION-TEST SUBSETS
    train_loader, validation_loader, test_loader = build_train_valid_test_subsets()

    if resnet:
        net = ResNet18()
    else:
        net = MyNet()

    print('######### Network created #########')
    print('Architecture:\n', net)
    train_model(net, train_loader, validation_loader, test_loader)
