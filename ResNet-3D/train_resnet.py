from torch.utils.data import DataLoader
import torch.optim as optim
from dataloader import HscnnDataset
import pickle
import os
import torch
import pandas as pd
import logging
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from plot_metric.functions import BinaryClassification
from model_resnet3d import ResidualBlock3D, ResNet3D
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score, recall_score, plot_confusion_matrix,\
    roc_auc_score
# Transforms
from torchio.transforms import (
    RandomFlip,
    RandomAffine,
    RandomMotion,
    RescaleIntensity,
    Compose,
)

LOG_FILENAME = 'C:\\Users\\AnilYadav\\PycharmProjects\\ResNet3D-UNet\\ResNet-3D\\ResNet3D-LOG_FOLDS.log'
logging.basicConfig(filename=LOG_FILENAME, level=logging.INFO,
                    format='%(levelname)s: %(message)s')


class TrainModel:

    def __init__(self, input_channel=1, malignancy_class=2, num_epochs=300, batch_size=6):
        self.in_channel = input_channel
        self.malignancy_class = malignancy_class
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.device = self._get_device()

    def _get_device(self):
        train_on_gpu = torch.cuda.is_available()
        if not train_on_gpu:
            device = torch.device("cpu")
            logging.info("Running on {}".format(device))
        else:
            device = torch.device("cuda:0")
            logging.info("Running on {}".format(torch.cuda.get_device_name(0)))
        return device

    def _get_default_transforms(self):
        io_transforms = Compose([
            RandomMotion(),
            RandomFlip(axes=(1,)),
            RandomAffine(scales=(0.9, 1.2), degrees=(10), isotropic=False, default_pad_value='otsu',
                         image_interpolation='bspline'),
            RescaleIntensity((0, 1))
        ])
        return io_transforms

    def _weights_init(self, model):
        classname = model.__class__.__name__
        if classname.find('Conv3d') != -1:
            nn.init.xavier_uniform_(model.weight, gain=nn.init.calculate_gain('relu'))
            model.bias.data.fill_(0)

    def _class_weights(self, path_to_label):
        data = pd.read_csv(path_to_label)
        weights = {}
        lbl = data.iloc[:, -1].value_counts()
        weight_0 = lbl[1] / (lbl[0] + lbl[1])
        weight_1 = lbl[0] / (lbl[0] + lbl[1])
        weights = [weight_0, weight_1]
        return weights

    def _get_index(self, input_index, num_fold):
        return input_index % num_fold

    def _load_pickle(self, input_file):
        with open(input_file, "rb") as f:
            pkl_file = pickle._Unpickler(f)
            pkl_file.encoding = 'latin1'
            pkl_file = pkl_file.load()
            return pkl_file

    def _merge_training_folds(self, path_to_training_fold, train_1_idx, train_2_idx):
        path_to_save = os.path.join(path_to_training_fold, "folds_merged")
        if not os.path.exists(path_to_save):
            os.makedirs(path_to_save)
        # Set up paths to training folds
        path_to_train_1 = path_to_training_fold + "fold_{}\\fold_{}_img.pkl".format(train_1_idx, train_1_idx)
        path_to_lbl_1 = path_to_training_fold + "fold_{}\\labels.csv".format(train_1_idx)
        path_to_train_2 = path_to_training_fold + "fold_{}\\fold_{}_img.pkl".format(train_2_idx, train_2_idx)
        path_to_lbl_2 = path_to_training_fold + "fold_{}\\labels.csv".format(train_2_idx)
        # Merge data and label
        img_1 = self._load_pickle(path_to_train_1)
        img_2 = self._load_pickle(path_to_train_2)
        new_img = np.concatenate((img_1, img_2), axis=0)
        lbl_1 = pd.read_csv(path_to_lbl_1)
        lbl_2 = pd.read_csv(path_to_lbl_2)
        frames = [lbl_1, lbl_2]
        df_concatenated = pd.concat(frames)

        # Save merged file and label
        pkl_name = "fold_{}_{}.pkl".format(train_1_idx, train_2_idx)
        full_path_lbl = os.path.join(path_to_save, "labels.csv")
        full_path_pkl = os.path.join(path_to_save, pkl_name)
        # Save Pickle file
        with open(full_path_pkl, 'wb') as file:
            pickle.dump(new_img, file)
        # Save csv file
        df_concatenated.to_csv(full_path_lbl, encoding='utf-8', index=False)
        return full_path_pkl, full_path_lbl

    def _get_evaluation_metric(self, y_truth, y_predicted, scores):
        evaluations = {}
        t_n, f_p, f_n, t_p = confusion_matrix(y_truth, y_predicted).ravel()
        # recall_sensitivity = t_p / (t_p + f_n)
        recall_sensitivity = recall_score(y_truth, y_predicted)
        specificity = t_n / (t_n + f_p)
        # accuracy = (t_p + t_n) / (f_p + f_n + t_p + t_n)
        accuracy = accuracy_score(y_truth, y_predicted)
        # precision = t_p / (t_p + f_p)
        precision = precision_score(y_truth, y_predicted)
        model_auc = roc_auc_score(y_truth, y_predicted)
        evaluations["Accuracy"] = accuracy
        evaluations["Sensitivity"] = recall_sensitivity
        evaluations["Specificity"] = specificity
        evaluations["Precision"] = precision
        evaluations["AUC"] = model_auc
        return evaluations

    def _start_model(self, model, optimizer, criterion, scheduler, train_loader, val_loader, test_loader, fold):
        if train_loader:
            train_losses, val_losses = [], []
            accuracy_list, specificity_list, recall_list = [], [], []
            precision_list, auc_list = [], []
            valid_loss_min = np.Inf
            for epoch in range(self.num_epochs):
                # Keep track of lossess
                running_train_loss, running_val_loss = 0.0, 0.0
                epoch_loss = []
                model.train()
                # Training loop
                for index, data in enumerate(train_loader):
                    images, labels = data['img'].to(self.device, dtype=torch.float), data['label'].to(self.device, dtype=torch.long)
                    optimizer.zero_grad()
                    output = model(images)
                    loss = criterion(output, labels)
                    loss.backward()
                    optimizer.step()
                    running_train_loss += float(loss.item()) * images.size(0)
                    epoch_loss.append(float(loss.item() * images.size(0)))

                scheduler.step(np.mean(epoch_loss))

                # Validation loop
                with torch.no_grad():
                    model.eval()
                    # Stores values for true and predicted labels
                    y_truth, y_predicted = [], []
                    for index, data in enumerate(val_loader):
                        images, labels = data['img'].to(self.device, dtype=torch.float), data['label'].to(self.device, dtype=torch.long)
                        output = model(images)
                        loss = criterion(output, labels)
                        running_val_loss += float(loss.item()) * images.size(0)
                        output_pb = F.softmax(output.cpu(), dim=1)
                        top_ps, top_class = output_pb.topk(1, dim=1)
                        y_predicted.extend(list(top_class.flatten().numpy()))
                        y_truth.extend(list(labels.cpu().flatten().numpy()))

                avg_train_loss = running_train_loss / len(train_loader)
                avg_val_loss = running_val_loss / len(val_loader)
                train_losses.append(avg_train_loss)
                val_losses.append(avg_val_loss)
                metrics = self._get_evaluation_metric(y_truth, y_predicted)
                accuracy_list.append(metrics['Accuracy'])
                specificity_list.append(metrics['Specificity'])
                recall_list.append(metrics['Sensitivity'])
                precision_list.append(metrics['Precision'])
                auc_list.append(metrics['AUC'])

                print("Epoch:{} - Training Loss:{:.6f} | Validation Loss: {:.6f}".format(epoch, avg_train_loss, avg_val_loss))
                # Save model if validation loss decreases
                if avg_val_loss <= valid_loss_min:
                    print("Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...".format(valid_loss_min,
                                                                                                    avg_val_loss))
                    print("-" * 40)
                    for arg in metrics:
                        print("{}: {}".format(arg, metrics[arg]))
                        logging.info("{}: {}".format(arg, metrics[arg]))
                    print("-" * 40)
                    logging.info("=" * 40)
                    # Save models
                    fold_wgt_path = "checkpoints/ResNet3D_{}.pth".format(fold)
                    torch.save(model.state_dict(), fold_wgt_path)
                    # Update minimum valdidation loss
                    valid_loss_min = avg_val_loss
                # Delete from memeory
                del metrics, y_truth, y_predicted, running_train_loss, running_val_loss

            # Saving Figures
            plt.plot(train_losses, label='Training loss')
            plt.plot(val_losses, label='Validation loss')
            plt.legend(frameon=False)
            plt.savefig('metrics_fold_{}_losses.png'.format(fold))
            plt.clf()

            plt.plot(recall_list, label='Sensitivity')
            plt.plot(specificity_list, label='Specificity')
            plt.legend(frameon=False)
            plt.savefig('metrics_fold_{}_sens_spec.png'.format(fold))
            plt.clf()

            plt.plot(accuracy_list, label='Accuracy')
            plt.legend(frameon=False)
            plt.savefig('metrics_fold_{}_accuracy.png'.format(fold))
            plt.clf()
            del train_losses, val_losses, accuracy_list, specificity_list, recall_list, precision_list, auc_list
            # Run TestSet
            self._runtestset(model, test_loader, fold_wgt_path, fold)

        else:
            raise ValueError('Model must be instantiated and Trainloader/Validation loader cannot be empty!')

    def _runtestset(self, model, test_loader, path_to_weights, fold):
        if path_to_weights:
            weights = torch.load(path_to_weights)
            model.load_state_dict(weights)
            model.to(self.device)
            # Make Predictions
            with torch.no_grad():
                model.eval()
                y_truth, y_predicted = [], []
                for data in test_loader:
                    images, labels = data['image'].to(self.device, dtype=torch.float), data['label'].to(self.device, dtype=torch.long)
                    output = model(images)
                    output_pb = F.softmax(output.cpu(), dim=1)
                    top_ps, top_class = output_pb.topk(1, dim=1)
                    y_predicted.extend(list(top_class.flatten().numpy()))
                    y_truth.extend(list(labels.cpu().flatten().numpy()))

                metrics = self._get_evaluation_metric(y_truth, y_predicted)
                logging.info("Test_Fold_{}_Metrics".format(fold))
                for arg in metrics:
                    print("{}: {}".format(arg, metrics[arg]))
                    logging.info("{}: {}".format(arg, metrics[arg]))
                logging.info("=" * 40)

                bc = BinaryClassification(y_truth, y_predicted, labels=["Benign", "Malignant"])
                bc.plot_roc_curve()
                auc_name = "Test_fold_{}_roc.png".format(fold)
                plt.savefig(auc_name)
                plt.clf()

    def start_training(self, path_to_training_folds=None, num_folds=(3,7)):
        if path_to_training_folds and num_folds:
            fold = 0
            for index in range(num_folds[0], num_folds[1]):
                # Set up path for folds
                test_idx, train_1_idx, train_2_idx, val_idx = self._get_index(index, 4), self._get_index(index + 1, 4), \
                                                              self._get_index(index + 2, 4), self._get_index(index + 3,
                                                                                                             4)

                print("Training fold: {}".format(fold + 1))
                logging.info("Training fold: {}".format(fold + 1))
                print("Test Index: {}, Train Index:{}, Val Index:{}".format(test_idx, (train_1_idx, train_2_idx),
                                                                            val_idx))
                # Test set and labels
                path_to_test_fold = os.path.join(path_to_training_folds,
                                                 'fold_{}\\fold_{}_img.pkl'.format(test_idx, test_idx))
                path_to_test_lbl = os.path.join(path_to_training_folds, 'fold_{}\\labels.csv'.format(test_idx))
                # Val set and label
                path_to_val_fold = os.path.join(path_to_training_folds,
                                                'fold_{}\\fold_{}_img.pkl'.format(val_idx, val_idx))
                path_to_val_lbl = os.path.join(path_to_training_folds, 'fold_{}\\labels.csv'.format(val_idx))
                # Get the merged training set with labels
                path_to_train_fold, path_to_train_lbl = self._merge_training_folds(path_to_training_folds,
                                                                                   train_1_idx, train_2_idx)

                lbl_weights = torch.FloatTensor(self._class_weights(path_to_train_lbl)).to(self.device)
                # Set up Dataset
                train_dataset = HscnnDataset(path_to_train_fold, path_to_train_lbl, transform=self._get_default_transforms())
                val_dataset = HscnnDataset(path_to_val_fold, path_to_val_lbl, transform=self._get_default_transforms())
                test_dataset = HscnnDataset(path_to_test_fold, path_to_test_lbl, transform=None)
                # Set up Dataloader
                train_loader = DataLoader(train_dataset, batch_size=self.batch_size, num_workers=3, shuffle=True, drop_last=True)
                val_loader = DataLoader(val_dataset, batch_size=self.batch_size, num_workers=3, shuffle=True, drop_last=True)
                test_loader = DataLoader(test_dataset, batch_size=self.batch_size, num_workers=3, shuffle=True, drop_last=True)

                # Instantiate the model
                model = ResNet3D(ResidualBlock3D, [2,2,2,2], in_channel, malignancy_class).to(self.device)
                optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
                criterion = nn.CrossEntropyLoss(weight=lbl_weights)

                # Begin training
                self._start_model(model, optimizer, criterion, scheduler, train_loader, val_loader, test_loader, fold)
                fold += 1
                del train_loader, val_loader, test_loader, model, optimizer, scheduler, criterion
                # Delete the merged files
                os.remove(path_to_train_fold)
                os.remove(path_to_train_lbl)
                logging.info("*" * 40)
                return None
        else:
            return "Path to Training Fold is required!"


if __name__ == "__main__":
    # Hyper-param
    number_folds = (3, 7)
    in_channel = 1
    malignancy_class = 2
    num_epochs = 1
    batch_size = 5
    # Call for Training
    path_to_folds = "path_to_training_data (in pickle_format)"
    train_obj = TrainModel(in_channel, malignancy_class, num_epochs, batch_size)
    train_obj.start_training(path_to_folds, number_folds)