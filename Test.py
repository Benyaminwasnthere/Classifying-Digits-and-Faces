import os
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import transforms, utils
from tensorflow.keras.utils import Progbar
from torch.utils.data import Dataset, DataLoader
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

# Enables benchmark mode in cudnn.
torch.backends.cudnn.benchmark = True

class TextDataset(Dataset):
    def __init__(self, data_text_path, data_labels_path, save_path, image_size, transforms=None):
        super().__init__()
        
        self.save_path = save_path
        self.transforms = transforms
        
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
            
            with open(data_text_path, 'r') as f:
                all_imgs = [list(map(int, line.rstrip('\n').replace(' ', '0 ').replace('#', '255 ').replace('+', '128 ').split())) for line in f]
            all_imgs = np.asarray(all_imgs, np.uint8).reshape((-1, *image_size))
            
            for i, img in enumerate(all_imgs):
                np.save(f'{save_path}/{i}.npy', img)

        with open(data_labels_path, 'r') as f:
            self.data_labels = np.asarray([int(line.rstrip('\n')) for line in f], dtype=np.uint8)
                
    def __len__(self):
        return len(self.data_labels)
    
    def __getitem__(self, index):
        if isinstance(index, slice):
            index = list(range(0 if index.start is None else index.start, self.__len__() if index.stop is None else index.stop, 1 if index.step is None else index.step))
            imgs = np.asarray([np.load(f'{self.save_path}/{i}.npy') for i in index], dtype=np.uint8)
            sample = (imgs, self.data_labels[index])
        elif isinstance(index, list) or isinstance(index, np.ndarray):
            imgs = np.asarray([np.load(f'{self.save_path}/{i}.npy') for i in index], dtype=np.uint8)
            sample = (imgs, self.data_labels[index])
        else:
            img = np.load(f'{self.save_path}/{index}.npy')
            sample = (img, self.data_labels[index])
        
        if self.transforms:
            sample = self.transforms(sample)
        
        return sample
    
digits_train_ds = TextDataset('./data/digitdata/trainingimages', './data/digitdata/traininglabels', './data/digitdata/train_data', image_size=(28, 28))
digits_val_ds = TextDataset('./data/digitdata/validationimages', './data/digitdata/validationlabels', './data/digitdata/val_data', image_size=(28, 28))
digits_test_ds = TextDataset('./data/digitdata/testimages', './data/digitdata/testlabels', './data/digitdata/test_data', image_size=(28, 28))

faces_train_ds = TextDataset('./data/facedata/facedatatrain', './data/facedata/facedatatrainlabels', './data/facedata/train_data', image_size=(70, 60))
faces_val_ds = TextDataset('./data/facedata/facedatavalidation', './data/facedata/facedatavalidationlabels', './data/facedata/val_data', image_size=(70, 60))
faces_test_ds = TextDataset('./data/facedata/facedatatest', './data/facedata/facedatatestlabels', './data/facedata/test_data', image_size=(70, 60))

BATCH_SIZE = 64

digits_train_dataloader = DataLoader(digits_train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
digits_val_dataloader = DataLoader(digits_val_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

faces_train_dataloader = DataLoader(faces_train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
faces_val_dataloader = DataLoader(faces_val_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = "cpu"
print("Device:", device)


class NaiveBayesClassifier:
    def __init__(self, possible_feature_vals, device, num_classes=10, num_features=784, alpha=1.0):
        self.device = device
        self.num_classes = num_classes
        self.num_features = num_features
        self.alpha = alpha
        
        self.log_class_priors = torch.zeros(num_classes).to(self.device)
        self.possible_feature_vals = torch.tensor(possible_feature_vals, dtype=torch.int32).to(self.device)
        self.log_pixel_likelihoods = torch.zeros((num_classes, self.num_features, len(self.possible_feature_vals))).to(self.device)
        
    def train(self, train_loader, val_loader=None):
        '''
        Compute class priors and pixel likelihoods from training data.
        '''
        total_samples = 0
        class_counts = torch.zeros(self.num_classes).to(self.device)
        pixel_counts = torch.zeros((self.num_classes, self.num_features, len(self.possible_feature_vals))).to(self.device)
        
        progbar = Progbar(target=len(train_loader), stateful_metrics=[])
        metrics = {}
        progbar.update(0, values=metrics.items(), finalize=False)

        train_ds = train_loader.dataset[:]
        if val_loader:
            val_ds = val_loader.dataset[:]

        for step, (images, labels) in enumerate(train_loader):
            images, labels = images.to(self.device), labels.to(self.device)
            total_samples += len(labels)
            # Each entry in the class_counts array stores the count of the number of times a particular class occurs throughout the dataset.
            class_counts += torch.sum(F.one_hot(labels.long(), num_classes=self.num_classes), dim=0)
            
            # Update pixel counts for each class where each feature pixel matches with a particular feature value.
            for c in range(self.num_classes):
                indices = torch.where(labels == c)
                pixel_counts[c, :, :] += torch.sum(images[indices].reshape(-1, self.num_features, 1) == self.possible_feature_vals, dim=0)

            # Update the class priors and pixel likelihoods after each iteration (so that validation receives the latest updated values).
            # The log values are pre-computed for efficiency.
            # The class priors are estimated as the probability of a random image in the dataset belonging to a particular class.
            self.log_class_priors = torch.log(class_counts + self.alpha) - np.log(total_samples + self.alpha * self.num_classes)
            # The pixel likelihoods are estimated.
            self.log_pixel_likelihoods = torch.log(pixel_counts + self.alpha) - torch.log(class_counts.reshape((-1, 1, 1)) + self.alpha * len(self.possible_feature_vals))

            metrics.update({'train_acc': self.predict(train_ds[0], train_ds[1], save_class_probs=True)[1]})

            if val_loader:
                metrics.update({'val_acc': self.predict(val_ds[0], val_ds[1])[1]})

            progbar.update(step, values=metrics.items(), finalize=False)

        progbar.update(step + 1, values=metrics.items(), finalize=True)
    
    def predict(self, images, gt_labels=None, save_class_probs=False):
        '''
        Compute class probabilities for each image in the batch using maximum a posteriori estimation.
        '''
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)
        images = images.reshape(-1, self.num_features, 1).to(self.device)
        
        # Compute the mask tensor.
        mask = (images == self.possible_feature_vals).to(torch.float32)
        
        # Perform element-wise multiplication of mask and self.pixel_likelihoods to extract the relevant probabilities.
        likelihoods = torch.sum(torch.sum(mask[:, None, :, :] * self.log_pixel_likelihoods[None, :, :, :], dim=-1), dim=-1)
        
        # Compute the log class probabilities.
        class_probs = self.log_class_priors + likelihoods

        if save_class_probs:
            self.class_probs = torch.exp(class_probs)

        preds = class_probs.argmax(dim=1)
        
        # Return the class predictions and accuracies.
        if gt_labels is not None: return preds, (preds == torch.from_numpy(gt_labels).to(self.device)).sum().item() / images.shape[0]

        return preds
    
    def generate_image(self, target_class, image_size):
        pixel_values = np.zeros((self.num_features,), dtype=np.int32)

        for i in range(self.num_features):
            log_probs = self.log_pixel_likelihoods[target_class, i, :]
            probs = torch.exp(log_probs - torch.max(log_probs)) # Compute the probabilities by taking the exponential of the log probabilities and normalizing.
            probs = probs.cpu().numpy()
            probs /= np.sum(probs)
            # Sample a value for the pixel from the possible values using the model's learned pixel probability distribution.
            pixel_values[i] = np.random.choice(self.possible_feature_vals.cpu().numpy(), p=probs)

        # Reshape the pixel values to form the image.
        image = pixel_values.reshape(image_size)
    
        return image
    


nb_models = []
portion = 0.1

for i in range(10):
    digits_train_dataloader = DataLoader(torch.utils.data.Subset(digits_train_ds, list(range(int(len(digits_train_ds) * portion)))), 
                                         batch_size=min(BATCH_SIZE, int(len(digits_train_ds) * portion)), 
                                         shuffle=True, 
                                          
                                         drop_last=False)
    
    nb_model = NaiveBayesClassifier(possible_feature_vals=[0, 128, 255], device=device)
    print(f'Training on {int(round(portion * 100))}% of the training data...')
    nb_model.train(digits_train_dataloader, digits_val_dataloader)
    nb_models.append(nb_model)
    
    portion += 0.1