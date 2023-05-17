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
import csv
from datetime import datetime
import time
import os

assert torch.cuda.is_available()

criterion = nn.CrossEntropyLoss().cuda()

logging.basicConfig(level=logging.INFO, format='%(message)s')


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        # hyperparameters
        input_size = 784
        output_size = 10
        hidden_size = 500

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


class CondenseAndTest:
    def __init__(self, density, no_compress):
        self.density = density
        self.no_compress = no_compress
        self.file_num = str(density)
        self.file_num = self.file_num.replace('.', '')
        self.results_path = f'condensa_outputs/{self.file_num}-results'

        # makes the results diectory if it doesn't exist
        if not os.path.exists(self.results_path):
            os.makedirs(self.results_path)

        self.csv_data = {}
        self.shap_data = {}

        self.csv_data['TimeStamp'] = (datetime.now().strftime("%d/%m/%Y %H:%M:%S"))

        self.test_loader, self.test_loader_unshuffled, self.train_loader, self.val_loader = self.get_data()

        self.base_model = Network()
        self.base_model.load_state_dict(torch.load('/home/yoshisada/Desktop/condensa/notebooks/model.pth'))
        self.condensed_model = None
        self.compressed_shap_vals = []

        self.condensa_filename = '/home/yoshisada/Desktop/condensa/notebooks/condensa_outputs/master_pruning_data.csv'
        self.shap_vals_filename = '/home/yoshisada/Desktop/condensa/notebooks/condensa_outputs/shap_vals.csv'
        self.shap_metrics_filename = '/home/yoshisada/Desktop/condensa/notebooks/condensa_outputs/shap_metrics.csv'


    '''
    Performs the full test stack
    '''
    def runTheGuantlet(self):
        self.condenseNeuron()
        self.runShap()
        self.countZeroWeights()
        self.testAccuracy()
        self.end()

    '''
    Condenses the models
    '''
    def evaluateSHAP(self):
        return

    def explainBase(self):
        # since shuffle=True, this is a random sample of test data
        batch = next(iter(self.test_loader))
        images, _ = batch

        background = images[:100]
        test_images = images[100:103]

        e = shap.DeepExplainer(self.condensed_model, background)
        shap_values = e.shap_values(test_images)

        print(np.array(shap_values).shape)

        return shap_values


    '''
    End the Test
    '''
    def end(self):
        condensa_fields = ['PruningDensity', 'PruningType', 'Accuracy', 'NonZeros', 'TimeStamp', 'CompressionTime']

        with open(self.condensa_filename, 'a') as csvfile: 
            # creating a csv dict writer object 
            writer = csv.DictWriter(csvfile, fieldnames = condensa_fields) 

            # writing singular row
            writer.writerow(self.csv_data)

        # clear in order to make sure row data is from same run
        self.csv_data.clear()

    '''
    write to the shap CSV
    '''
    def write_to_shap(self):
        shap_fields = ['PruningDensity', 'PruningType', 'Actual', 'Predicted', 'Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine']

        with open(self.shap_vals_filename, 'a') as csvfile: 
            # creating a csv dict writer object 
            writer = csv.DictWriter(csvfile, fieldnames = shap_fields) 

            # writing singular row
            writer.writerow(self.shap_data)

        # clear in order to make sure row data is from same run
        self.csv_data.clear()
    



    '''
    Setup the data loaders for condensa
    Used to inialize the class' loaders
    '''
    def get_data(self):
        # local size values for getting the data
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

        # used to get the same shap values everytime
        test_loader_unshuffled = DataLoader(mnist_test, batch_size=batch_size_test, shuffle=False)
        test_loader = DataLoader(mnist_test, batch_size=batch_size_test, shuffle=True)
        train_loader = DataLoader(train_split, batch_size=batch_size_train, shuffle=True)
        val_loader = DataLoader(val_split, batch_size=BATCH_SIZE, shuffle=True)

        return test_loader, test_loader_unshuffled, train_loader, val_loader

    '''
    Run Condensa
    '''

    # TODO: uncommnet the compression logic
    def condenseNeuron(self):
        if (self.no_compress):
            return

        self.csv_data['PruningDensity'] = self.density
        self.csv_data['PruningType'] = 'neuron'

        self.shap_data['PruningDensity'] = self.density
        self.shap_data['PruningType'] = 'neuron'

        NEURON = condensa.schemes.NeuronPrune(self.density)

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
                                            self.base_model,
                                            self.train_loader,
                                            self.test_loader,
                                            self.val_loader,
                                            criterion)

        # Run the compression and time how long it takes
        start_time = time.time()
        w_neuron = compressor_NEURON.run()
        completion_time = time.time() - start_time

        self.csv_data['CompressionTime'] = round(completion_time, 3)

        torch.save(w_neuron.state_dict(), f'{self.results_path}/MNIST_NEURON_{self.file_num}.pth')

        self.condensed_model = Network()
        self.condensed_model.load_state_dict(torch.load(f'/home/yoshisada/Desktop/condensa/notebooks/{self.results_path}/MNIST_NEURON_{self.file_num}.pth'))
    

    '''
    Run SHAP on the Compressed Model
    '''
    def runShap(self):
        # since shuffle=True, this is a random sample of test data
        batch = next(iter(self.test_loader))
        images, _ = batch

        background = images[:100]
        test_images = images[100:103]

        e = shap.DeepExplainer(self.condensed_model, background)
        shap_values = e.shap_values(test_images)
        self.compressed_shap_vals = shap_values

        shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
        test_numpy = np.swapaxes(np.swapaxes(test_images.numpy(), 1, -1), 1, 2)

        shap.image_plot(shap_numpy, -test_numpy, show=False)
        plt.savefig(f'{self.results_path}/shap{self.file_num}.png')  
        
    '''
    SHAP with percentages, done on specific images
    '''

    # TODO make it easier to figure out what number you want to inspect
    def runShapWithPercentages(self):
        images, targets = next(iter(self.test_loader_unshuffled))

        NUM_SHIFT = 7
        BACKGROUND_SIZE = 100 + NUM_SHIFT
        background_images = images[:BACKGROUND_SIZE]
        background_targets = targets[:BACKGROUND_SIZE].numpy()

        # TODO: limit test_images and target here to the number i want
        SELECTED_IMAGES = BACKGROUND_SIZE + 1

        test_images = images[BACKGROUND_SIZE:SELECTED_IMAGES]
        test_targets = targets[BACKGROUND_SIZE:SELECTED_IMAGES].numpy()

        numToText = {0:'Zero', 1:'One', 2:'Two', 3:'Three', 4:'Four', 5:'Five', 6:'Six', 7:'Seven', 8:'Eight', 9:'Nine'}
        
        # TODO: get rid of the for loop somehow 
        def show_attributions(model):
            # Predict the probabilities of the digits using the test images
            output = model(test_images)
            # Get the index of the max log-probability
            pred = output.max(1, keepdim=True)[1] 
            # Convert to numpy only once to save time
            pred_np = pred.numpy() 

            expl = shap.DeepExplainer(model, background_images)

            
            for i in range(0, len(test_images)):
                ti = test_images[[i]]
                sv = expl.shap_values(ti)
                self.compressed_shap_vals = sv
                sn = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in sv]
                tn = np.swapaxes(np.swapaxes(ti.numpy(), 1, -1), 1, 2)

                # Prepare the attribution plot, but do not draw it yet
                # We will add more info to the plots later in the code
                shap.image_plot(sn, -tn, show=False)

                # Prepare to augment the plot
                fig = plt.gcf()
                allaxes = fig.get_axes()

                # Show the actual/predicted class
                allaxes[0].set_title('Actual: {}, pred: {}'.format(
                    test_targets[i], pred_np[i][0]))
                
                self.shap_data['Actual'] = test_targets[i]
                self.shap_data['Predicted'] = pred_np[i][0]

                # Show the probability of each class
                # There are 11 axes for each picture: 1 for the digit + 10 for each SHAP
                # There is a last axis for the scale - we don't want to apply a label for that one
                prob = output[i].detach().numpy()
                
                # put all the info in the correct csv label
                for x in range(1, len(allaxes)-1):
                    self.shap_data[numToText[x-1]] = round(prob[x-1] * 100, 2)
            
                # add percentage labels
                for x in range(1, len(allaxes)-1):
                    allaxes[x].set_title('{:.2%}'.format(prob[x-1]), fontsize=14)
                
                # store the image in the right place
                # TODO: make sure it writes name corresponding to chosen num to inspect
                # plt.savefig(f'{self.results_path}/shap_with_percentages_num_one_{self.file_num}.png')
                plt.savefig(f'/home/yoshisada/Desktop/condensa/notebooks/num_1_shap_images/shap_with_percentages_num_one_{self.file_num}.png')

        if (self.condensed_model is not None):
            show_attributions(self.condensed_model)
        elif (os.path.isfile(f'{self.results_path}/MNIST_NEURON_{self.file_num}.pth')):
            condensed_model = Network()
            condensed_model.load_state_dict(torch.load(f'/home/yoshisada/Desktop/condensa/notebooks/{self.results_path}/MNIST_NEURON_{self.file_num}.pth'))
            show_attributions(condensed_model)
        else:
            return

    def shapSumOfSquares(self, base_shap_vals):
        shap_shape = np.array(base_shap_vals).shape
        sum = 0
        for label in range(shap_shape[0]):
            for pixels in range(shap_shape[-2]):
                for pixel in range(shap_shape[-1]):
                    sqrd_diff = pow((base_shap_vals[label][0][0][pixels][pixel] -
                                        self.compressed_shap_vals[label][0][0][pixels][pixel]),
                                    2)
                    sum += sqrd_diff

        print(sum)
        # metric_fields = ['Metric', 'Density', 'Output']
        # sum_of_sqrs_data = {
        #     'Metric': 'Sum of Squares',
        #     'Density': self.density,
        #     'Output': sum
        # }

        # with open(self.shap_metrics_filename, 'a') as csvfile: 
        #     # creating a csv dict writer object 
        #     writer = csv.DictWriter(csvfile, fieldnames = metric_fields) 

        #     # writing singular row
        #     writer.writerow(sum_of_sqrs_data)

        # # clear in order to make sure row data is from same run
        # sum_of_sqrs_data.clear()


    '''
    Count the Number of Non-Zero Parameters
    '''
    def countZeroWeights(self):
        non_zeros = 0
        for param in self.condensed_model.parameters():
            if param is not None:
                non_zeros += torch.sum((param != 0).int()).item()

        # condensa_out.write(f'Number of non-zeros on PRUNED model: {condensed_count}')
        self.csv_data['NonZeros'] = non_zeros
        print('Number of non-zeros on PRUNED model: ', non_zeros)

    '''
    Get the Accuracy of the Compressed Model
    '''
    def testAccuracy(self):
        test_losses = []
        self.condensed_model.eval()
        test_loss = 0
        correct = 0
        accuracy = 0
        with torch.no_grad():
            for data, target in self.test_loader:
            #   data = data.reshape(-1, 28*28)
                output = self.condensed_model(data)
                test_loss += F.nll_loss(output, target, size_average=False).item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()
        test_loss /= len(self.test_loader.dataset)
        test_losses.append(test_loss)
        # writes to csv
        accuracy = '{:.0f}'.format(100. * correct / len(self.test_loader.dataset))
        self.csv_data['Accuracy'] = accuracy
        # condensa_out.write('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        #     test_loss, correct, len(test_loader.dataset),
        #     100. * correct / len(test_loader.dataset)))
        