import datetime
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt

from training.models import MLP
from preprocessing.clean_data import get_data
from training.utils import validate_routine

# HYPERPARAMETERS
num_epochs = 200
batch_size = 64
lr = 1e-1
momentum = 0.
dropout = 0.5
activation = F.relu
dims = [256, 128]
test = True

# LOAD TRAINING DATA
train_data, validation_data, test_data = get_data(batch_size, from_file=True)

# LOAD MODEL AND OPTIMIZER
model = MLP(2, 300, 512, *dims, dropout=dropout, activation=activation)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

# TRAINING LOOP
train_ls = []
train_accs = []
val_accs = []

for ep in range(num_epochs):
    train_loss = 0
    train_acc = 0
    model.train()

    for inputs, labels in train_data:
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        train_loss += loss.item()
        train_acc += (outputs.argmax(1) == labels).sum().item()

        loss.backward()
        optimizer.step()

        optimizer.zero_grad()

    # Validation
    val_acc, positives, true_positives, positive_labeled = validate_routine(model, validation_data)

    # Log results
    train_ls.append(train_loss / len(train_data))
    train_accs.append(train_acc / len(train_data.dataset))
    val_accs.append(val_acc / len(validation_data.dataset))

    if ep > 1:
        print(f'epoch {ep}, train loss {train_loss / len(train_data)}, '
              f'train accuracy {train_acc / len(train_data.dataset)}, '
              f'val accuracy {val_acc / len(validation_data.dataset)}, '
              f'val precision {true_positives / positives}, '
              f'val recall {true_positives / positive_labeled}')

dt = str(datetime.datetime.now())
dt = dt[:dt.index('.')].replace(' ', '_')
torch.save(model, f'model_{dt}.pt')

# PLOTS
fig, axs = plt.subplots(2)
axs[0].plot(train_accs, color='blue', label='Training')
axs[0].plot(val_accs, color='green', label='Validation')
axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('Accuracy')
axs[0].legend()

axs[1].plot(train_ls, color='orange')
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('Training Loss')
acc = round(val_acc / len(validation_data.dataset) * 100, 3)
axs[0].set_title(f'Final Validation Accuracy: {acc} %')
plt.show()

if test:
    test_acc = validate_routine(model, test_data)[0]
    test_total = len(test_data.dataset)
    print(f'\n Test Results: got {test_acc} of {test_total} correct, i.e. '
          f'{round(test_acc / test_total * 100, 3)} % test accuracy !!!')
