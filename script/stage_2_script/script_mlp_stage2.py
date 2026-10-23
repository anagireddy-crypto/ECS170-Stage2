import csv
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

template_root = Path(__file__).resolve().parents[2]
if str(template_root) not in sys.path:
    sys.path.insert(0, str(template_root))

from local_code.stage_2_code.Dataset_Loader_Stage2 import Dataset_Loader_Stage2
from local_code.stage_2_code.Evaluate_Classification_Stage2 import Evaluate_Classification_Stage2
from local_code.stage_2_code.Method_MLP_Stage2 import Method_MLP_Stage2


random_seed = 170
number_of_epochs = 8
batch_size = 256
learning_rate = 0.001

random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)

result_folder = template_root / 'result' / 'stage_2_result'
result_folder.mkdir(parents=True, exist_ok=True)

data_loader = Dataset_Loader_Stage2()
data = data_loader.load()

print('Training examples:', len(data['train']['y']))
print('Testing examples:', len(data['test']['y']))
print('Input features:', data['train']['X'].shape[1])
print('Classes:', len(np.unique(data['train']['y'])))

train_x = torch.tensor(data['train']['X'], dtype=torch.float32)
train_y = torch.tensor(data['train']['y'], dtype=torch.long)
test_x = torch.tensor(data['test']['X'], dtype=torch.float32)
training_data = TensorDataset(train_x, train_y)

experiments = [
    {'name': 'baseline_128', 'hidden_size': 128, 'second_hidden_size': 0},
    {'name': 'two_layers_128_64', 'hidden_size': 128, 'second_hidden_size': 64},
    {'name': 'wider_256', 'hidden_size': 256, 'second_hidden_size': 0},
]

all_losses = []
all_results = []
evaluator = Evaluate_Classification_Stage2()

for experiment in experiments:
    print('\nExperiment:', experiment['name'])

    # Use the same seed so each model starts from a repeatable initialization.
    torch.manual_seed(random_seed)
    model = Method_MLP_Stage2(
        hidden_size=experiment['hidden_size'],
        second_hidden_size=experiment['second_hidden_size'],
    )

    shuffle_generator = torch.Generator().manual_seed(random_seed)
    train_loader = DataLoader(
        training_data, batch_size=batch_size, shuffle=True, generator=shuffle_generator
    )

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    epoch_losses = []

    for epoch in range(number_of_epochs):
        model.train()
        total_loss = 0.0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = loss_function(predictions, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_y)

        average_loss = total_loss / len(training_data)
        epoch_losses.append(average_loss)
        all_losses.append([experiment['name'], epoch + 1, average_loss])
        print('Epoch', epoch + 1, 'of', number_of_epochs, '- loss:', round(average_loss, 4))

    model.eval()
    with torch.no_grad():
        test_predictions = model(test_x).argmax(dim=1).numpy()

    evaluator.data = {'true_y': data['test']['y'], 'pred_y': test_predictions}
    scores = evaluator.evaluate()
    result = {
        'experiment': experiment['name'],
        'hidden_layers': str(experiment['hidden_size'])
        if experiment['second_hidden_size'] == 0
        else str(experiment['hidden_size']) + ',' + str(experiment['second_hidden_size']),
        **scores,
    }
    all_results.append(result)
    print('Test scores:', scores)

    plt.plot(range(1, number_of_epochs + 1), epoch_losses, marker='o', label=experiment['name'])

plt.title('Stage 2 MNIST training loss')
plt.xlabel('Epoch')
plt.ylabel('Average training loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(result_folder / 'training_loss.png', dpi=150)
plt.close()

with (result_folder / 'training_losses.csv').open('w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['experiment', 'epoch', 'average_training_loss'])
    writer.writerows(all_losses)

with (result_folder / 'experiment_results.csv').open('w', newline='') as file:
    columns = ['experiment', 'hidden_layers', 'accuracy', 'weighted_precision', 'weighted_recall', 'weighted_f1']
    writer = csv.DictWriter(file, fieldnames=columns)
    writer.writeheader()
    writer.writerows(all_results)

with (result_folder / 'experiment_settings.csv').open('w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['random_seed', 'epochs', 'batch_size', 'learning_rate', 'optimizer', 'loss_function'])
    writer.writerow([random_seed, number_of_epochs, batch_size, learning_rate, 'Adam', 'CrossEntropyLoss'])

print('\nSaved training plot and experiment results to:', result_folder)
