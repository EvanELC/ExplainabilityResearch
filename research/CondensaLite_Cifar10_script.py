import torch
import torch.nn as nn
import logging
from datetime import datetime
import os
import shap
import csv
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader, Subset
from ExplainabilityResearch import condensa
import time 
import json
import numpy as np
import matplotlib.pyplot as plt

assert torch.cuda.is_available()
criterion = nn.CrossEntropyLoss().cuda()

logging.basicConfig(level=logging.INFO, format='%(message)s')

import torch.nn as nn


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.features = self._make_layers([64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
    
class CondensaLiteCifar10:
    def __init__(self, density, no_compress):
        self.density = density
        self.no_compress = no_compress
        self.file_num = str(density)
        self.file_num = self.file_num.replace('.', '')
        self.results_path = f'cifar10_outputs/{self.file_num}-results'
        self.scheme_path = "/home/yoshisada/Desktop/ExplainabilityResearch/research/schemes/prune.json"

        # makes the results diectory if it doesn't exist
        if not os.path.exists(self.results_path):
            os.makedirs(self.results_path)

        self.csv_data = {}

        self.csv_data['TimeStamp'] = (datetime.now().strftime("%d/%m/%Y %H:%M:%S"))

        self.test_loader, self.test_loader_unshuffled, self.train_loader, self.val_loader = self.get_data()

        self.base_model = Network()

        """ special stuff do clean up the model to load it in will have to be done for any model from: https://github.com/kuangliu/pytorch-cifar""" 
        # Define the file path of the checkpoint
        checkpoint_path = '/home/yoshisada/Desktop/CIFAR10_Models/pytorch-cifar/checkpoint/VGG19_92ACC.pth'

        # Load the checkpoint file
        checkpoint = torch.load(checkpoint_path)

        # remove modeule prefix on values
        remove_prefix = 'module.'
        state_dict = {k[len(remove_prefix):] if k.startswith(remove_prefix) else k: v for k, v in checkpoint["net"].items()}
        """ end of special stuff to load model"""

        # Load the model
        self.base_model.load_state_dict(state_dict)
        self.condensed_model = None
        self.condensa_filename = '/home/yoshisada/Desktop/ExplainabilityResearch/research/cifar10_outputs/master_pruning_data.csv'
 
        # specify the image classes
        self.classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']
    
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
    Gets the SHAP values from the base model and returns it
    '''
    def explainBase(self):
        # since shuffle=True, this is a random sample of test data
        batch = next(iter(self.test_loader))
        images, _ = batch

        background = images[:100]
        test_images = images[100:101]

        e = shap.DeepExplainer(self.condensed_model, background)
        shap_values = e.shap_values(test_images)

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
    **changed to cifar10 dataloaders**

    Setup the data loaders for condensa
    Used to inialize the class' loaders
    '''
    def get_data(self):
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        VAL_SIZE = 0.1
        BATCH_SIZE = 64
        batch_size_train = 64
        batch_size_test = 1000

        cifar10_train = torchvision.datasets.CIFAR10('/home/yoshisada/Desktop/ExplainabilityResearch/research/DataSet_Files', train=True, download=True,
                             transform=transform_train)

        cifar10_test = torchvision.datasets.CIFAR10('/home/yoshisada/Desktop/ExplainabilityResearch/research/DataSet_Files', train=False, download=True,
                                    transform=transform_test)
        
        # generate indices: instead of the actual data we pass in integers instead
        train_indices, val_indices, _, _ = train_test_split(
            range(len(cifar10_train)),
            cifar10_train.targets,
            stratify=cifar10_train.targets,
            test_size=VAL_SIZE,
        )

        # generate subset based on indices
        train_split = Subset(cifar10_train, train_indices)
        val_split = Subset(cifar10_train, val_indices)

        # create batches
        # used to get the same shap values everytime
        test_loader_unshuffled = DataLoader(cifar10_test, batch_size=batch_size_test, shuffle=False)
        train_loader = DataLoader(train_split, batch_size=batch_size_train, shuffle=True)
        val_loader = DataLoader(val_split, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(cifar10_test, batch_size=batch_size_test, shuffle=True)

        return test_loader, test_loader_unshuffled, train_loader, val_loader
    
    '''
    Run Condensa
    '''
    def condenseNeuron(self):
        if (self.no_compress):
            return

        self.csv_data['PruningDensity'] = self.density
        self.csv_data['PruningType'] = 'unstructured'

        """ change the desisty of the scheme file json"""
        # Load JSON file
        with open(self.scheme_path, 'r') as file:
            data = json.load(file)

        # Change the sparsity value to 0.2
        data['sparsity'] = self.density

        # Write the modified data back to the JSON file
        with open(self.scheme_path, 'w') as file:
            json.dump(data, file, indent=4)
        """ END """

        modules = []
        for name, m in self.base_model.named_modules():
            if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv2d):
                modules.append(name)
    
        # Run the compression and time how long it takes
        # TODO: base_model gets over written here
        start_time = time.time()
        scheme = condensa.schemes.parse(self.scheme_path, modules)
        with condensa.save_masks():
            scheme.pi(self.base_model)
        completion_time = time.time() - start_time

        self.csv_data['CompressionTime'] = round(completion_time, 3)

        torch.save(self.base_model.state_dict(), f'{self.results_path}/CIFAR10_unstructured_{self.file_num}.pth')

        self.condensed_model = Network()
        self.condensed_model.load_state_dict(torch.load(f'/home/yoshisada/Desktop/ExplainabilityResearch/research/{self.results_path}/CIFAR10_unstructured_{self.file_num}.pth'))

    '''
    Run SHAP on the Compressed Model
    makes the image pretty with labels and predictions
    '''
    def runShap(self):
        images, targets = next(iter(self.test_loader))
        BACKGROUND_SIZE = 100
        background_images = images[:BACKGROUND_SIZE]
        
        # TODO: limit test_images and target here to the number i want
        test_images = images[BACKGROUND_SIZE:]
        test_targets = targets[BACKGROUND_SIZE:].numpy()

        # create the shap images
        def show_attributions(model, how_many = 1):
            # Predict the probabilities of the digits using the test images
            output = model(test_images)
            # Get the index of the max log-probability
            pred = output.max(1, keepdim=True)[1] 
            # Convert to numpy only once to save time
            pred_np = pred.numpy() 

            expl = shap.DeepExplainer(model, background_images)

            for i in range(0, how_many):
                ti = test_images[[i]]
                sv = expl.shap_values(ti)
                sn = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in sv]
                tn = np.swapaxes(np.swapaxes(ti.numpy(), 1, -1), 1, 2)

                # Prepare the attribution plot, but do not draw it yet
                # We will add more info to the plots later in the code
                shap.image_plot(sn, -tn, width=30, show=False)

                # Prepare to augment the plot
                fig = plt.gcf()
                allaxes = fig.get_axes()

                # Show the actual/predicted class
                allaxes[0].set_title('Actual: {}\npred: {}'.format(
                    self.classes[test_targets[i]], self.classes[pred_np[i][0]]))

                # Show the probability of each class
                # There are 11 axes for each picture: 1 for the digit + 10 for each SHAP
                # There is a last axis for the scale - we don't want to apply a label for that one
                prob = output[i].detach().numpy()
                
                for x in range(1, len(allaxes)-1):
                    allaxes[x].set_title('{}\n{:.2%}'.format(self.classes[x-1], prob[x-1]), fontsize=14)

                plt.savefig(f'{self.results_path}/shap{self.file_num}.png')
        
        show_attributions(self.condensed_model)

    
    #TODO: add call to this function
    '''
    Calculates the sum of squares of the SHAP values for each
    pixel between the uncompressed and compressed models
    '''
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
        train_on_gpu = torch.cuda.is_available()
        BATCH_SIZE = 64

        # track test loss
        test_loss = 0.0
        class_correct = list(0. for i in range(10))
        class_total = list(0. for i in range(10))

        self.condensed_model.eval()
        # iterate over test data
        for data, target in self.test_loader:
            # move tensors to GPU if CUDA is available
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()
                '''
                fix to speed up accuracy calculation
                move model to the gpu 
                '''
                self.condensed_model = self.condensed_model.cuda()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = self.condensed_model(data)
            # calculate the batch loss
            loss = criterion(output, target)
            # update test loss 
            test_loss += loss.item()*data.size(0)
            # convert output probabilities to predicted class
            _, pred = torch.max(output, 1)    
            # compare predictions to true label
            correct_tensor = pred.eq(target.data.view_as(pred))
            correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
            # calculate test accuracy for each object class
            for i in range(BATCH_SIZE):
                label = target.data[i]
                class_correct[label] += correct[i].item()
                class_total[label] += 1

        '''
        move model back to cpu to avoid any issues
        '''
        self.condensed_model = self.condensed_model.cpu()

        # average test loss
        test_loss = test_loss/len(self.test_loader.dataset)
        print('Test Loss: {:.6f}\n'.format(test_loss))

        for i in range(10):
            if class_total[i] > 0:
                print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                    self.classes[i], 100 * class_correct[i] / class_total[i],
                    np.sum(class_correct[i]), np.sum(class_total[i])))
            else:
                print('Test Accuracy of %5s: N/A (no training examples)' % (self.classes[i]))

        print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
            100. * np.sum(class_correct) / np.sum(class_total),
            np.sum(class_correct), np.sum(class_total)))

        accuracy = '{:.0f}'.format(100. * np.sum(class_correct) / np.sum(class_total))
        print('acc value:', accuracy)
        self.csv_data['Accuracy'] = accuracy
