import torch
import torch.nn as nn
from condensa.schemes import NeuronPrune
import logging
import condensa
import shap
import numpy as np
import torchvision
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import getopt, sys
from condensa.schemes import Compose, Prune, Quantize
import util
import torchvision.datasets as datasets
import torch.nn.functional as F
import matplotlib.pyplot as plt

assert torch.cuda.is_available()

criterion = nn.CrossEntropyLoss().cuda()

logging.basicConfig(level=logging.INFO, format='%(message)s')


'''
Get input prune parameter
'''
# opts, args = getopt.getopt(sys.argv[1:])
prune_val = float(sys.argv[1])
out_file_num = sys.argv[2]

RESULTS_PATH = f'condensa_outputs/{out_file_num}-results'

condensa_out = open(f'{RESULTS_PATH}/{out_file_num}-pruning_stats.txt', 'a')

##########

'''
Setup the data loaders for condensa
'''
VAL_SIZE = 0.1
BATCH_SIZE = 64
batch_size_train = 64
batch_size_test = 1000

mnist_train = torchvision.datasets.MNIST('/home/yoshisada/Desktop/condensa/notebooks/files', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ]))

mnist_test = torchvision.datasets.MNIST('/home/yoshisada/Desktop/condensa/notebooks/files', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ]))

train_indices, val_indices, _, _ = train_test_split(
    range(len(mnist_train)),
    mnist_train.targets,
    stratify=mnist_train.targets,
    test_size=VAL_SIZE,
)

train_split = Subset(mnist_train, train_indices)
val_split = Subset(mnist_train, val_indices)

test_loader = DataLoader(mnist_test, batch_size=batch_size_test, shuffle=True)
train_loader = DataLoader(train_split, batch_size=batch_size_train, shuffle=True)
val_loader = DataLoader(val_split, batch_size=BATCH_SIZE, shuffle=True)

################

'''
Create the network
'''
# hyperparameters
input_size = 784
output_size = 10
hidden_size = 500

class Network(nn.Module):
    
    def __init__(self):
        super(Network, self).__init__()
        self.flatten = nn.Flatten()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = self.flatten(x)
        x = self.l1(x)
        x = self.relu(x)
        x = self.l3(x)
        return F.log_softmax(x, dim = 1)
    
network = Network()
network.load_state_dict(torch.load('/home/yoshisada/Desktop/condensa/notebooks/model.pth'))

###########
'''
Run Condensa
'''

NEURON = condensa.schemes.NeuronPrune(prune_val)

lc = condensa.opt.LC(steps=35,                             # L-C iterations
                     l_optimizer=condensa.opt.lc.SGD,      # L-step sub-optimizer
                     l_optimizer_params={'momentum':0.95}, # L-step sub-optimizer parameters
                     lr=0.01,                              # Initial learning rate
                     lr_end=1e-4,                          # Final learning rate
                     mb_iterations_per_l=3000,             # Mini-batch iterations per L-step
                     mb_iterations_first_l=30000,          # Mini-batch iterations for first L-step
                     mu_init=1e-3,                         # Initial value of `mu`
                     mu_multiplier=1.1,                    # Multiplier for `mu`
                     mu_cap=10000,                         # Maximum value of `mu`
                     debugging_flags={'custom_model_statistics':
                                      condensa.util.cnn_statistics})
compressor_NEURON = condensa.Compressor(lc,
                                      NEURON,
                                      network,
                                      train_loader,
                                      test_loader,
                                      val_loader,
                                      criterion)
w_neuron = compressor_NEURON.run()

torch.save(w_neuron.state_dict(), f'{RESULTS_PATH}/MNIST_NEURON_{out_file_num}.pth')

##########
'''
Run SHAP on the condensed model
'''

# load model in from memory
condensed_model = Network()
condensed_model.load_state_dict(torch.load(f'/home/yoshisada/Desktop/condensa/notebooks/{RESULTS_PATH}/MNIST_NEURON_{out_file_num}.pth'))


# since shuffle=True, this is a random sample of test data
batch = next(iter(test_loader))
images, _ = batch

background = images[:100]
test_images = images[100:103]

print(len(background), len(test_images))

e = shap.DeepExplainer(condensed_model, background)
shap_values = e.shap_values(test_images)

shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
test_numpy = np.swapaxes(np.swapaxes(test_images.numpy(), 1, -1), 1, 2)

shap.image_plot(shap_numpy, -test_numpy, show=False)
plt.savefig(f'{RESULTS_PATH}/shap{out_file_num}.png')

##########
'''
Count the paramters in the model
'''
def countZeroWeights(model):
    zeros = 0
    for param in model.parameters():
        if param is not None:
            zeros += torch.sum((param != 0).int()).item()
    return zeros

condensed_count = countZeroWeights(condensed_model)
condensa_out.write(f'Number of non-zeros on PRUNED model: {condensed_count}')
print('Number of non-zeros on PRUNED model: ', condensed_count)


##########
'''
Test the accuracy of the network
'''
test_losses = []
def test(model):
  model.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
    #   data = data.reshape(-1, 28*28)
      output = model(data)
      test_loss += F.nll_loss(output, target, size_average=False).item()
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum()
  test_loss /= len(test_loader.dataset)
  test_losses.append(test_loss)
  # writes to output
  condensa_out.write('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))
  print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))
  
test(condensed_model)

condensa_out.close()
