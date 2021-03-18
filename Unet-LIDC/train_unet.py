from build_unet_model import *
from custom_loss import *
from inferences import _get_iou_vector
from dataloader_unet import DataProcessor
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from torch.autograd import Variable
import torch
import torch.nn as nn
import numpy as np
from default_tranforms import RandomNoise, VerticalFlip, HorizontalFlip, ToTensor


class TrainModel:
    def __init__(self, image_channel, num_out_classes, num_epochs, batch_size, weights_path):
        self.image_channel = image_channel
        self.num_out_classes = num_out_classes
        self.epochs = num_epochs
        self.batch = batch_size
        self.path_to_weights = weights_path
        self.device = self._get_device()

    def _get_device(self):
        train_on_gpu = torch.cuda.is_available()
        if not train_on_gpu:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:0")
        return device

    def _get_default_transforms(self):
        my_transforms = transforms.Compose([transforms.ToPILImage(), transforms.RandomHorizontalFlip(p=0.5), transforms.ColorJitter(brightness=0.5),
                                            transforms.RandomCrop((224, 224)), transforms.RandomRotation(degrees=45), transforms.RandomVerticalFlip(p=0.2),
                                            transforms.ToTensor()])
        return my_transforms

    def start_training(self, path_to_train_images, path_to_train_masks, path_to_valid_images,
                       path_to_valid_masks, transformation=None, lr_rate=1e-4):
        if transformation is None:
            transformations_train = transforms.Compose([RandomNoise(), VerticalFlip(), HorizontalFlip(), ToTensor()])
            transformations_valid = transforms.Compose([ToTensor()])

        train_dataset = DataProcessor(imgs_dir=path_to_train_images, masks_dir=path_to_train_masks, transformation=transformations_train)
        valid_dataset = DataProcessor(imgs_dir=path_to_valid_images, masks_dir=path_to_valid_masks, transformation=transformations_valid)

        print("="*40)
        print("Images for Training:", len(train_dataset))
        print("Images for Validation:", len(valid_dataset))
        print("="*40)

        trainloader = DataLoader(train_dataset, batch_size=self.batch, shuffle=True, drop_last=True)
        validloader = DataLoader(valid_dataset, batch_size=self.batch, shuffle=True, drop_last=True)

        """
        data = iter(trainloader).next()
        image, mask = data["img"], data["mask"]
        print("image shape: {}, min-max: {}".format(image.shape, (image.min(), image.max())))
        print("mask shape: {}, min-max: {}".format(mask.shape, (mask.min(), mask.max())))
        print("=" * 40)
        
        # For visualizing nodule
        for arg in range(image.shape[0]):
            mask_data = np.array(mask[arg, 0])
            rgb_image = np.array(image[arg]).transpose((1,2,0))
            label_image = label(mask_data)
            image_label_overlay = label2rgb(label_image, image=rgb_image, bg_label=0, kind='overlay')
            fig, axs = plt.subplots(nrows=1, ncols=3)
            axs[0].imshow(rgb_image, cmap="gray")
            axs[1].imshow(mask_data, cmap="gray")
            axs[2].imshow(image_label_overlay, cmap="gray")
            plt.show()
        """

        # Instantiate model and other parameters
        model = UNet(self.image_channel, self.num_out_classes).to(self.device)
        # Load Weights if available
        if path_to_weights:
            weights = torch.load(path_to_weights, map_location=self.device)
            model.load_state_dict(weights)
        # Define three losses
        criterion1 = nn.BCEWithLogitsLoss().to(self.device)
        criterion2 = SoftDiceLoss().to(self.device)
        criterion3 = InvSoftDiceLoss().to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
        # Varibles to track
        running_bce, running_dice, running_invtdice, running_loss_comb = [], [], [], []
        val_bce, val_dice, val_invtdice, val_loss_comb, ious = [], [], [], [], []
        global_avg_iou = 0.0

        for epoch in range(self.epochs):
            running_bce_loss, running_dice_loss, running_invtdice_loss, running_train_loss = 0.0, 0.0, 0.0, 0.0
            val_bce_loss, val_dice_loss, val_invtdice_loss, val_loss = 0.0, 0.0, 0.0, 0.0
            epoch_loss, avg_iou = [], 0.0
            model.train()
            for i, train_data in enumerate(trainloader):
                images, masks = train_data["img"], train_data["mask"]
                images = Variable(images).to(self.device)
                masks = Variable(masks).to(self.device)
                optimizer.zero_grad()
                outputs = model(images)
                bce_loss, dice_loss, invt_dice_loss = criterion1(outputs, masks), criterion2(outputs, masks), criterion3(outputs, masks)
                loss = bce_loss + dice_loss + invt_dice_loss
                loss.backward()
                optimizer.step()
                # Track train loss
                running_bce_loss += float(bce_loss.item()) * images.size(0)
                running_dice_loss += float(dice_loss.item()) * images.size(0)
                running_invtdice_loss += float(invt_dice_loss.item()) * images.size(0)
                running_train_loss += float(loss.item()) * images.size(0)
                epoch_loss.append(float(loss.item() * images.size(0))) # For scheluder

            scheduler.step(np.mean(epoch_loss))
            with torch.no_grad():
                model.eval()
                for valid_data in validloader:
                    images, masks = valid_data["img"], valid_data["mask"]
                    images = Variable(images).to(self.device)
                    masks = Variable(masks).to(self.device)
                    outputs = model(images)
                    output_prob = torch.sigmoid(outputs).detach().cpu().numpy()
                    output_gt = masks.detach().cpu().numpy()
                    output_prob_thresh = (output_prob > 0.5) * 1
                    avg_iou += _get_iou_vector(output_gt, output_prob_thresh)
                    # Calculate Losses
                    bce_loss, dice_loss, invt_dice_loss = criterion1(outputs, masks), criterion2(outputs, masks), criterion3(outputs, masks)
                    loss = bce_loss + dice_loss + invt_dice_loss
                    # Track val loss
                    val_bce_loss += float(bce_loss.item()) * images.size(0)
                    val_dice_loss += float(dice_loss.item()) * images.size(0)
                    val_invtdice_loss += float(invt_dice_loss.item()) * images.size(0)
                    val_loss += float(loss.item()) * images.size(0)

            # Average the metrics
            avg_iou_batch = avg_iou / len(validloader) # IoU
            # Average train metrics
            avg_train_bce_loss = running_bce_loss / len(trainloader)
            avg_train_dice_loss = running_dice_loss / len(trainloader)
            avg_train_invtdice_loss = running_invtdice_loss / len(trainloader)
            avg_train_loss = running_train_loss / len(trainloader)

            # Average Val metrics
            avg_val_bce_loss = val_bce_loss / len(validloader)
            avg_val_dice_loss = val_dice_loss / len(validloader)
            avg_val_invtdice_loss = val_invtdice_loss / len(validloader)
            avg_val_loss_combined = val_loss / len(validloader)

            # Append metrics for tracking
            running_bce.append(avg_train_bce_loss)
            running_dice.append(avg_train_dice_loss)
            running_invtdice.append(avg_train_invtdice_loss)
            running_loss_comb.append(avg_train_loss)
            val_bce.append(avg_val_bce_loss)
            val_dice.append(avg_val_dice_loss)
            val_invtdice.append(avg_val_invtdice_loss)
            val_loss_comb.append(avg_val_loss_combined)
            ious.append(avg_iou_batch)

            print("Epoch {}, Training Loss(BCE+Dice+InvtDice): {}, Validation Loss(BCE): {}, Validation Loss(Dice): {}, Validation Loss(InvDice): {} Average IoU: {}".format(epoch + 1, avg_train_loss,
                                                                                                                                                                             avg_val_bce_loss, avg_val_dice_loss,
                                                                                                                                                                             avg_val_invtdice_loss, avg_iou_batch))

            if avg_iou_batch > global_avg_iou:
                print("Average mask IoU increased: ({:.6f} --> {:.6f}).  Saving model ...".format(global_avg_iou,
                                                                                                  avg_iou_batch))
                print("-" * 40)
                # Save model
                torch.save(model.state_dict(), 'checkpoints/LungMask_UNet.pth')
                global_avg_iou = avg_iou_batch

        # Save Plots - Train Unet
        plt.plot(running_bce, label='BCE loss')
        plt.plot(running_dice, label='Dice loss')
        plt.plot(running_invtdice, label='Invt Dice loss')
        plt.plot(running_loss_comb, label='Combined loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(frameon=False)
        plt.savefig('train_losses.png')
        plt.clf()

        # Val loss
        plt.plot(val_bce, label='BCE loss')
        plt.plot(val_dice, label='Dice loss')
        plt.plot(val_invtdice, label='Invt Dice loss')
        plt.plot(val_loss_comb, label='Combined loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(frameon=False)
        plt.savefig('val_losses.png')
        plt.clf()

        # IoU
        plt.plot(ious, label='Mask IoU')
        plt.xlabel('Epoch')
        plt.ylabel('IoU')
        plt.legend(frameon=False)
        plt.savefig('val_iou.png')
        plt.clf()


if __name__ == "__main__":
    # Hyper-param
    img_channel = 3
    out_num_class = 1
    num_epcohs = 10
    batches = 2
    path_to_weights = None
    train_images = "path_to_training_slices"
    train_masks = "path_to_training_masks"
    valid_images = "path_to_validation_slcies"
    valid_masks = "path_to_validation_masks"
    train_obj = TrainModel(img_channel, out_num_class, num_epcohs, batches, path_to_weights)
    train_obj.start_training(train_images, train_masks, valid_images, valid_masks)
    #train_obj.runtestset(test_images)
