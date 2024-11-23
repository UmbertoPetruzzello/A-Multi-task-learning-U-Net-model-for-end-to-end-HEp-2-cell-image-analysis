import pathlib
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from torch.nn.modules import loss
from torchsummary import summary
import torchvision
from data_new import HEP2Dataset
from model_new import UNet
from loss import MultiTaskLoss
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import csv
import pandas as pd
import os
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import imutils
from numpy import newaxis
import sklearn.metrics as sk
import cv2
import random
from sklearn.utils import shuffle

#writer = SummaryWriter()
parser = argparse.ArgumentParser(description='Common training and evaluation.')
parser.add_argument('--da', dest='da', type=bool, default=False, help='Apply or not DA')
parser.add_argument('--pp', dest='pp', type=bool, default=False, help='Apply or not Pre-Process')
parser.add_argument('--dim_x', dest='dimx', type=int, default=384, help='Dimension X of the patch')
parser.add_argument('--dim_y', dest='dimy', type=int, default=384, help='Dimension Y of the patch')
parser.add_argument('--stride_x', dest='strx', type=int, default=251, help='Stride X among the patches')
parser.add_argument('--stride_y', dest='stry', type=int, default=328, help='Stride Y among the patches')
parser.add_argument('--patch_n', dest='patchn', type=int, default=15, help='Number of the patches')
args = parser.parse_args()

def predict(img, model, preprocess, device):
    model.eval()
    if args.pp:
        img = preprocess(img)  # preprocess image
    img = img.to(device)
    with torch.no_grad():
        seg, lab, intens = model(img)  # send through model/network
    return seg, lab, intens

def dice_score(mask_1, mask_2):
    volume_sum = mask_1.sum() + mask_2.sum() + 1e-8
    volume_intersect = (mask_1 & mask_2).sum()
    return 2*volume_intersect / volume_sum

def most_frequent(List):
    counter = 0
    num = List[0]
    for i in List:
        curr_frequency = List.count(i)
        if(curr_frequency> counter):
            counter = curr_frequency
            num = i
    return num 
    
def preprocess(im):
    minim = np.amin(im)
    maxim = np.amax (im)
    diff = (maxim - minim)
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            im[i][j] = (im[i][j] - minim) / diff
    return im

#patch reconstruction function - dependent to the patch dimension.
def patch_reconstruction(mask):
    result = np.zeros((1040, 1388), np.uint8)
    for y in range(1388):
        for x in range(1040):
            if x < 328:
                if y < 251:
                    result[x][y] = masks[0][x][y]
                elif y < 384:
                    s = float(masks[0][x][y]) + float(masks[3][x][y - 251])
                    if s >= 255:
                        result[x][y] = 255
                    else: 
                        result[x][y] = 0
                elif y < 502:
                    result[x][y] = masks[3][x][y - 251]
                elif y < 635:
                    s = float(masks[6][x][y - 502]) + float(masks[3][x][y - 251])
                    if s >= 255:
                        result[x][y] = 255
                    else: 
                        result[x][y] = 0
                elif y < 753:
                    result[x][y] = masks[6][x][y - 502]
                elif y < 886:
                    s = float(masks[6][x][y - 502]) + float(masks[9][x][y - 753])
                    if s >= 255:
                        result[x][y] = 255
                    else: 
                        result[x][y] = 0
                elif y < 1004:
                    result[x][y] = masks[9][x][y - 753]
                elif y < 1137:
                    s = float(masks[12][x][y - 1004]) + float(masks[9][x][y - 753])
                    if s >= 255:
                        result[x][y] = 255
                    else: 
                        result[x][y] = 0
                else: #x>=1137
                    result[x][y] = masks[12][x][y - 1004]
                    
            elif x < 384:
                if y < 251:
                    s = float(masks[0][x][y]) + float(masks[1][x - 328][y])
                    if s >= 255:
                        result[x][y] = 255
                    else: 
                        result[x][y] = 0
                elif y < 384:
                    s = float(masks[0][x][y]) + float(masks[3][x][y - 251]) + float(masks[1][x - 328][y]) + float(masks[4][x - 328][y - 251])
                    if s >= 510:
                        result[x][y] = 255
                    else: 
                        result[x][y] = 0
                elif y < 502:
                    s = float(masks[3][x][y - 251]) + float(masks[4][x - 328][y - 251])
                    if s >= 255:
                        result[x][y] = 255
                    else: 
                        result[x][y] = 0
                elif y < 635:
                    s = float(masks[3][x][y - 251]) + float(masks[6][x][y - 502]) + float(masks[4][x - 328][y - 251]) + float(masks[7][x - 328][y - 502])
                    if s >= 510:
                        result[x][y] = 255
                    else: 
                        result[x][y] = 0
                elif y < 753:
                    s = float(masks[6][x][y - 502]) + float(masks[7][x - 328][y - 502]) 
                    if s >= 255:
                        result[x][y] = 255
                    else: 
                        result[x][y] = 0
                elif y < 886:
                    s = float(masks[6][x][y - 502]) + float(masks[9][x][y - 753]) + float(masks[7][x - 328][y - 502]) + float(masks[10][x - 328][y - 753])
                    if s >= 510:
                        result[x][y] = 255
                    else: 
                        result[x][y] = 0
                elif y < 1004:
                    s = float(masks[9][x][y - 753]) + float(masks[10][x - 328][y - 753]) 
                    if s >= 255:
                        result[x][y] = 255
                    else: 
                        result[x][y] = 0
                elif y < 1137:
                    s = float(masks[9][x][y - 753]) + float(masks[12][x][y - 1004]) + float(masks[10][x - 328][y - 753]) + float(masks[13][x - 328][y - 1004])
                    if s >= 510:
                        result[x][y] = 255
                    else: 
                        result[x][y] = 0
                else: #x>=1137
                    s = float(masks[12][x][y - 1004]) + float(masks[13][x - 328][y - 1004]) 
                    if s >= 255:
                        result[x][y] = 255
                    else: 
                        result[x][y] = 0
                    
            elif x < 656:
                if y < 251:
                    result[x][y] = masks[1][x - 328][y]
                elif y < 384:
                    s = float(masks[1][x - 328][y]) + float(masks[4][x - 328][y - 251])
                    if s >= 255:
                        result[x][y] = 255
                    else: 
                        result[x][y] = 0
                elif y < 502:
                    result[x][y] = masks[4][x - 328][y - 251]
                elif y < 635:
                    s = float(masks[4][x - 328][y - 251]) + float(masks[7][x - 328][y - 502])
                    if s >= 255:
                        result[x][y] = 255
                    else: 
                        result[x][y] = 0
                elif y < 753:
                    result[x][y] = masks[7][x - 328][y - 502]
                elif y < 886:
                    s = float(masks[7][x - 328][y - 502]) + float(masks[10][x - 328][y - 753])
                    if s >= 255:
                        result[x][y] = 255
                    else: 
                        result[x][y] = 0
                elif y < 1004:
                    result[x][y] = masks[10][x - 328][y - 753]
                elif y < 1137:
                    s = float(masks[13][x - 328][y - 1004]) + float(masks[10][x - 328][y - 753])
                    if s >= 255:
                        result[x][y] = 255
                    else: 
                        result[x][y] = 0
                else: #x>=1137
                    result[x][y] = masks[13][x - 328][y - 1004]
                
            elif x < 712:
                if y < 251:
                    s = float(masks[1][x - 328][y]) + float(masks[2][x - 656][y]) 
                    if s >= 255:
                        result[x][y] = 255
                    else: 
                        result[x][y] = 0
                elif y < 384:
                    s = float(masks[1][x - 328][y]) + float(masks[4][x - 328][y - 251]) + float(masks[2][x - 656][y]) + float(masks[5][x - 656][y - 251])
                    if s >= 510:
                        result[x][y] = 255
                    else: 
                        result[x][y] = 0
                elif y < 502:
                    s = float(masks[4][x - 328][y - 251]) + float(masks[5][x - 656][y - 251])
                    if s >= 255:
                        result[x][y] = 255
                    else: 
                        result[x][y] = 0
                elif y < 635:
                    s = float(masks[4][x - 328][y - 251]) + float(masks[7][x - 328][y - 502]) + float(masks[5][x - 656][y - 251]) + float(masks[8][x - 656][y - 502])
                    if s >= 510:
                        result[x][y] = 255
                    else: 
                        result[x][y] = 0
                elif y < 753:
                    s = float(masks[7][x - 328][y - 502]) + float(masks[8][x - 656][y - 502])
                    if s >= 255:
                        result[x][y] = 255
                    else: 
                        result[x][y] = 0
                elif y < 886:
                    s = float(masks[7][x - 328][y - 502]) + float(masks[10][x - 328][y - 753]) + float(masks[8][x - 656][y - 502]) + float(masks[11][x - 656][y - 753])
                    if s >= 510:
                        result[x][y] = 255
                    else: 
                        result[x][y] = 0
                elif y < 1004:
                    s = float(masks[10][x - 328][y - 753]) + float(masks[11][x - 656][y - 753])
                    if s >= 255:
                        result[x][y] = 255
                    else: 
                        result[x][y] = 0
                elif y < 1137:
                    s = float(masks[10][x - 328][y - 753]) + float(masks[13][x - 328][y - 1004]) + float(masks[11][x - 656][y - 753]) + float(masks[14][x - 656][y - 1004])
                    if s >= 510:
                        result[x][y] = 255
                    else: 
                        result[x][y] = 0
                else: #x>=1137
                    s = float(masks[13][x - 328][y - 1004]) + float(masks[14][x - 656][y - 1004])
                    if s >= 255:
                        result[x][y] = 255
                    else: 
                        result[x][y] = 0
                
            else: #y>=712
                if y < 251:
                    result[x][y] = masks[2][x - 656][y]
                elif y < 384:
                    s = float(masks[2][x - 656][y]) + float(masks[5][x - 656][y - 251])
                    if s >= 255:
                        result[x][y] = 255
                    else: 
                        result[x][y] = 0
                elif y < 502:
                    result[x][y] = masks[5][x - 656][y - 251]
                elif y < 635:
                    s = float(masks[5][x - 656][y - 251]) + float(masks[8][x - 656][y - 502])
                    if s >= 255:
                        result[x][y] = 255
                    else: 
                        result[x][y] = 0
                elif y < 753:
                    result[x][y] = masks[8][x - 656][y - 502]
                elif y < 886:
                    s = float(masks[8][x - 656][y - 502]) + float(masks[11][x - 656][y - 753])
                    if s >= 255:
                        result[x][y] = 255
                    else: 
                        result[x][y] = 0
                elif y < 1004:
                    result[x][y] = masks[11][x - 656][y - 753]
                elif y < 1137:
                    s = float(masks[14][x - 656][y - 1004]) + float(masks[11][x - 656][y - 753])
                    if s >= 255:
                        result[x][y] = 255
                    else: 
                        result[x][y] = 0
                else: #x>=1137
                    result[x][y] = masks[14][x - 656][y - 1004]
    return result

class Trainer:
    def __init__(self,
                 model: torch.nn.Module,
                 device: torch.device,
                 criterion: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 training_DataLoader: torch.utils.data.Dataset,
                 validation_DataLoader: torch.utils.data.Dataset = None,
                 lr_scheduler: torch.optim.lr_scheduler = None,
                 epochs: int = 100,
                 epoch: int = 0,
                 notebook: bool = False,
                 fold = 1
                 ):

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.training_DataLoader = training_DataLoader
        self.validation_DataLoader = validation_DataLoader
        self.device = device
        self.epochs = epochs
        self.epoch = epoch
        self.notebook = notebook
        self.fold = fold
        
        self.minimum_valid_loss = 100
        self.training_loss = []
        self.validation_loss = []
        self.learning_rate = []

    def run_trainer(self):

        if self.notebook:
            from tqdm.notebook import tqdm, trange
        else:
            from tqdm import tqdm, trange

        progressbar = trange(self.epochs, desc='Progress')
        for i in progressbar:
            """Epoch counter"""
            self.epoch += 1  # epoch counter

            """Training block"""
            self._train()

            """Validation block"""
            if self.validation_DataLoader is not None:
                self._validate()

            """Learning rate scheduler block"""
            if self.lr_scheduler is not None:
                if self.validation_DataLoader is not None and self.lr_scheduler.__class__.__name__ == 'ReduceLROnPlateau':
                    self.lr_scheduler.batch(self.validation_loss[i])  # learning rate scheduler step with validation loss
                else:
                    self.lr_scheduler.batch()  # learning rate scheduler step

        return self.training_loss, self.validation_loss, self.learning_rate

    def _train(self):

        if self.notebook:
            from tqdm.notebook import tqdm, trange
        else:
            from tqdm import tqdm, trange

        self.model.train()  # train mode
        l = torch.as_tensor(0, dtype=torch.float32, device=device)
        train_losses = []  # accumulate the losses here
        c = 0
        batch_iter = tqdm(enumerate(self.training_DataLoader), 'Training', total=len(self.training_DataLoader),
                          leave=False)
        #with torch.profiler.profile(schedule=torch.profiler.schedule(wait=2, warmup=2, active=6, repeat=1), on_trace_ready=torch.profiler.tensorboard_trace_handler('./log2try/try'), with_stack=True) as profiler:
        for i, (x, y) in batch_iter:
            c += 1
            input = x.to(self.device)
            mask = y[0].to(self.device)
            lab = y[1].to(self.device)
            intens = y[2].to(self.device)
            
            self.optimizer.zero_grad(set_to_none=True)  # zerograd the parameters
            out = self.model(input)  # one forward pass
            loss = self.criterion(out, mask, lab, intens)  # calculate loss
            l += loss
            loss.backward()  # one backward pass
            self.optimizer.step()  # update the parameters
            #profiler.step()
            
            batch_iter.set_description(f'Training: ')
        
        #print(" TRAIN LOSS: ", np.mean(train_losses))
        losses = l.item()/c
        print("TRAIN LOSS: ", losses)  
        dice_loss, ce_loss, bce_loss = self.criterion.get_losses(c)
        print("Mean dice loss: ", dice_loss)
        print("Mean CE loss: ", ce_loss)
        print("Mean BCE loss: ", bce_loss)
        self.criterion.set_losses()
        self.training_loss.append(losses)
        self.learning_rate.append(self.optimizer.param_groups[0]['lr'])
        model_name = 'hep2_model_cross_v_' + self.fold + '.pt'
        torch.save(model.state_dict(), model_name)
        print("Model Saved")
        
        batch_iter.close()

    def _validate(self):

        if self.notebook:
            from tqdm.notebook import tqdm, trange
        else:
            from tqdm import tqdm, trange

        self.model.eval()  # evaluation mode
        valid_losses = []  # accumulate the losses here
        l = torch.as_tensor(0, dtype=torch.float32, device=device)
        batch_iter = tqdm(enumerate(self.validation_DataLoader), 'Validation', total=len(self.validation_DataLoader),
                          leave=False)
        c = 0
        for i, (x, y) in batch_iter:
            c += 1
            input = x.to(self.device)
            mask = y[0].to(self.device)
            lab = y[1].to(self.device)
            intens = y[2].to(self.device)

            with torch.no_grad():
                out = self.model(input)
                loss = self.criterion(out, mask, lab, intens)  # calculate loss
                l += loss

                batch_iter.set_description(f'Validation: ')
        
        losses = l.item()/c
        print(" VAL LOSS: ", losses)
        valid_mean_loss = losses
        dice_loss, ce_loss, bce_loss = self.criterion.get_losses(c)
        print("Mean dice loss: ", dice_loss)
        print("Mean CE loss: ", ce_loss)
        print("Mean BCE loss: ", bce_loss)
        self.criterion.set_losses()
        if valid_mean_loss < self.minimum_valid_loss:
            self.minimum_valid_loss = valid_mean_loss
            # save the model
            model_name = 'hep2_model_cross_v_' + self.fold + '.pt'
            torch.save(model.state_dict(), model_name)
            print("Model Saved ", model_name)
        self.validation_loss.append(losses)

        batch_iter.close()


# device
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

torch.manual_seed(42)

for i in range(100):
    model = UNet(1, 1)
    model = model.to(device)
    fold = str(i+1)
    if i == 0:
        summary = summary(model, (1, args.dimx, args.dimy))
    criterion = MultiTaskLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    root_dir = '/mnt/sdc1/upetruzzello/data/HEp-2_train_and_test/3_ICPR2014_Specimen_Dataset/datasets/train_ICPR2014_task2/train_patch/'
    frame_test_name = "/mnt/sdc1/upetruzzello/annotation/cross_fold" + fold +"_test.csv"
    frame_test = pd.read_csv(frame_test_name, names=["Image", "Mask", "Label", "Intensity"])
    train_string = "/mnt/sdc1/upetruzzello/annotation/cross_fold" + fold +"_train.csv"
    frame = pd.read_csv(train_string, names=["Image", "Mask", "Label", "Intensity"])

    train = HEP2Dataset(frame, root_dir, transform=args.da)
    
    trainloader = torch.utils.data.DataLoader(train, shuffle=True, batch_size=32, drop_last=True, pin_memory = True, num_workers=8)

    trainer = Trainer(model=model, device=device, criterion=criterion, optimizer=optimizer,
                      training_DataLoader=trainloader, validation_DataLoader=None, lr_scheduler=None,
                      epochs=10, epoch=0, notebook=False, fold = fold )
    
    training_losses, validation_losses, lr_rates = trainer.run_trainer()
    
    # testing
    print("Starting production of patch results")
    images = []
    ground = []
    names = []
    total_test = len(frame_test)
    for idx in range(total_test):
        img_name = os.path.join(root_dir, frame_test.iloc[idx, 0])
        image = io.imread(img_name)
        mask_name = os.path.join(root_dir, frame_test.iloc[idx, 1])
        mask = io.imread(mask_name)
        label = frame_test.iloc[idx, 2]
        intensity = frame_test.iloc[idx, 3]
        image = image[newaxis, newaxis, :, :]
        image = torch.from_numpy(image).type(torch.float32)
        images.append(image)
        ground.append((mask, label, intensity))
        names.append(frame_test.iloc[idx, 1])
    
    output = []
    f = 0
    intensity_prod = []
    label_prod = []
    label_tr = []
    intensity_tr = []
    for image in images:
        label_tr.append(ground[f][1])
        intensity_tr.append(ground[f][2])
        output.append(predict(image, model, preprocess, device))

        label = torch.softmax(output[f][1], dim = 1)
        label_prod.append(torch.argmax(label, 1).item())
        
        intensit = torch.sigmoid(output[f][2])
        intensity_prod.append(torch.where(intensit > 0.5, 1, 0).item())
        
        preds = torch.sigmoid(output[f][0])
        preds = (preds == 1.0).float()

        path_res = "produced_images/produced_mask_cross_val/" + names[f]

        torchvision.utils.save_image(preds, path_res)
        f += 1
        
    #calculate confusion matrix for label and  intensity
    lab_cm = sk.confusion_matrix(label_tr, label_prod)
    int_cm = sk.confusion_matrix(intensity_tr, intensity_prod)

    np.savetxt("csv_result/confusion_matrix_pattern_patches_" + str(fold) + ".csv", lab_cm, delimiter=';', fmt='%d')
    np.savetxt("csv_result/confusion_matrix_intensity_patches_" + str(fold) + ".csv", int_cm, delimiter=';', fmt='%d')

    path_of_confusion_matrix_patch = "csv_result/mean_class_accuracy_patches_" + str(fold) + ".txt"
    #calculate MCA for each label
    sum_tot = 0
    mca_partial = []
    for ii in range(len(lab_cm)):
        diag = lab_cm[ii][ii]
        sum = np.sum(lab_cm[ii])
        sum_tot += diag/sum
        mca_partial.append(diag/sum)
    MCA = sum_tot/7
    with open(path_of_confusion_matrix_patch, 'w') as fp:
        for element in mca_partial:
            fp.write(str(element))
            fp.write('\n')
    mca_partial = []
    print("***********MCA PATCHES***********: ", MCA)

    #calculate Accuracy for intensity
    dia = 0
    summ = 0
    for ii in range(len(int_cm)):
        diag += int_cm[ii][ii]
        sum += np.sum(int_cm[ii])
    ACC_INT = diag/sum
    print("***********ACCURACY PATCHES***********: ", ACC_INT)

    lab_cm = []
    int_cm = []
    output = []
    images = []
    ground = []
    
    root_dir_2 = 'produced_images/produced_mask_cross_val/'
    root_dir_fin = 'produced_images/compared_mask_cross_val/'

    result = np.zeros((args.dimx, args.dimy, 3), np.uint8)
    TOTAL_DICE = 0
    dice = 0
    TOTAL_ACC = 0
    acc = 0
    
    for idx in range(total_test):
        mask_name = os.path.join(root_dir, frame_test.iloc[idx, 1])
        string = names[idx]
        mask_pred_name = os.path.join(root_dir_2, string)
        mask = cv2.imread(mask_name, cv2.IMREAD_GRAYSCALE)
        mask_pred = cv2.imread(mask_pred_name, cv2.IMREAD_GRAYSCALE)
        c = 0
        for jj in range(args.dimx):
            for j in range(args.dimy):
                if mask_pred[jj][j] == mask[jj][j]:
                    c += 1
                    if mask_pred[jj][j] == 0:
                        result[jj][j] = (0,0,0)
                    else:
                        result[jj][j] = (255,255,255)
                else:
                    if mask_pred[jj][j] == 0:
                        result[jj][j] = (0,0,255) #maschera = 1 predetta = 0 (False Negative) - Red
                    else:
                        result[jj][j] = (0,255,255) #maschera = 0 predetta = 1 (False Positive) - Yellow
        #calculate dice and segmentation accuracy
        dice += dice_score(mask, mask_pred)
        acc += c/(args.dimx*args.dimy)
        #save images
        string_fin = root_dir_fin + string
        cv2.imwrite(string_fin, result)    
    TOTAL_DICE = dice / total_test
    TOTAL_ACC = acc / total_test
    print("***********DICE PATCHES***********: ",TOTAL_DICE)
    print("***********SEG.ACC. PATCHES***********: ",TOTAL_ACC)
    
    #append patch result to csv
    result_list_patch = [MCA, ACC_INT, TOTAL_DICE, TOTAL_ACC]
    
    with open('csv_result/patch_result_cross_val.csv', 'a', newline='') as f:
        csv_writer = csv.writer(f, delimiter=';')
        
        if i == 0:
            csv_writer.writerow(["Pattern Classification", "Intensity Classification", "Dice Score", "Segmentation Accuracy"])
        
        csv_writer.writerow(result_list_patch)
    result_list_patch = []
    
    
    #produce whole specimen results
    print("Starting production of whole specimen results")
    final_lab = []
    final_int = []
    final_lab_true = []
    final_int_true = []
    label_spec = []
    int_spec = []
    masks = []
    result = np.zeros((1040, 1388), np.uint8)
    for ii in range(total_test):
        label_spec.append(label_prod[ii])
        int_spec.append(intensity_prod[ii])
        if (ii + 1) % args.patchn == 0:
            q = int((ii)/args.patchn)
            start = args.patchn*q
            stop = args.patchn*(q+1)
            for k in reversed(range(args.patchn)):
                
                stringa_img = "produced_images/produced_mask_cross_val/" + names[ii - k]
                masks.append(cv2.imread(stringa_img, cv2.IMREAD_GRAYSCALE))
            
            result = patch_reconstruction(mask)
            fin_string = "produced_images/produced_whole_mask_cross_val/" + names[ii].split("_patch")[0] + "_Mask.tif"
            cv2.imwrite(fin_string, result) 
            result = np.zeros((1040, 1388), np.uint8)
            final_lab.append(most_frequent(labels_spec))
            final_int.append(most_frequent(int_spec))
            final_lab_true.append(label_tr[ii])
            final_int_true.append(intensity_tr[ii])
            labels_spec = []
            int_spec = []
            masks = []
            
    #calulate intensity on pattern
    count_h = 0
    count_s = 0
    count_n = 0
    count_c = 0
    count_nm = 0
    count_g = 0
    count_ms = 0
    iacc_h = 0
    iacc_s = 0
    iacc_n = 0
    iacc_c = 0
    iacc_nm = 0
    iacc_g = 0
    iacc_ms = 0
    
    for cc in range(len(final_int)):
        pred = final_int[i]
        int_lab = final_int_true[i]
        label_patt = final_lab_true[i]
        
        if label_patt == 0:
            label_to_check = 'homogeneous '
        elif label_patt == 1:
            label_to_check = 'speckled '
        elif label_patt == 2:
            label_to_check = 'nucleolar '
        elif label_patt == 3:
            label_to_check = 'centromere '
        elif label_patt == 4:
            label_to_check = 'golgi '
        elif label_patt == 5:
            label_to_check = 'numem '
        else:
            label_to_check = 'mistp '
        
        if label_to_check == 'homogeneous ':
                iacc_h += 1
                count_h += 1
        elif label_to_check == 'speckled ':
            iacc_s += 1
            count_s += 1
        elif label_to_check == 'nucleolar ':
            iacc_n += 1
            count_n += 1
        elif label_to_check == 'centromere ':
            iacc_c += 1
            count_c += 1
        elif label_to_check == 'numem ':
            iacc_nm += 1
            count_nm += 1
        elif label_to_check == 'golgi ':
            iacc_g += 1
            count_g += 1
        else:
            iacc_ms += 1
            count_ms += 1
                
    with open("csv_result/intensity_on_pattern.txt", 'w') as fint:
        fp.write(str(iacc_h))
        fp.write('\n')
        fp.write(str(count_h))
        fp.write('\n')
        fp.write(str(iacc_s))
        fp.write('\n')
        fp.write(str(count_s))
        fp.write('\n')
        fp.write(str(iacc_n))
        fp.write('\n')
        fp.write(str(count_n))
        fp.write('\n')
        fp.write(str(iacc_c))
        fp.write('\n')
        fp.write(str(count_c))
        fp.write('\n')
        fp.write(str(iacc_nm))
        fp.write('\n')
        fp.write(str(count_nm))
        fp.write('\n')
        fp.write(str(iacc_g))
        fp.write('\n')
        fp.write(str(count_g))
        fp.write('\n')
        fp.write(str(iacc_ms))
        fp.write('\n')
        fp.write(str(count_ms))
        fp.write('\n')
    
    #calculate confusion matrix for intensity of whole specimen

    lab_cm = sk.confusion_matrix(final_lab_true, final_lab)
    int_cm = sk.confusion_matrix(final_int_true, final_int)

    np.savetxt("csv_result/confusion_matrix_pattern_whole_" + str(fold) + ".csv", lab_cm, delimiter=';', fmt='%d')
    np.savetxt("csv_result/confusion_matrix_intensity_whole_" + str(fold) + ".csv", int_cm, delimiter=';', fmt='%d')

    path_of_confusion_matrix_whole = "csv_result/mean_class_accuracy_whole_" + str(fold) + ".txt"
    #calculate MCA for label
    sum_tot = 0
    mca_partial = []
    for ii in range(len(lab_cm)):
        diag = lab_cm[ii][ii]
        sum = np.sum(lab_cm[ii])
        sum_tot += diag/sum
        mca_partial.append(diag/sum)
    MCA_WHOLE = sum_tot/7
    with open(path_of_confusion_matrix_whole, 'w') as fp:
        for element in mca_partial:
            fp.write(str(element))
            fp.write('\n')
    mca_partial = []
    print("***********MCA SPECIMEN***********: ", MCA_WHOLE)
    
    #calculate Accuracy for intensity
    diag = 0
    sum = 0
    for ii in range(len(int_cm)):
        diag += int_cm[ii][ii]
        sum += np.sum(int_cm[ii])
    ACC_INT_WHOLE = diag/sum
    print("***********ACCURACY SPECIMEN***********: ", ACC_INT_WHOLE)

    final_lab = []
    final_int = []
    final_lab_true = []
    final_int_true = []
    lab_cm = []
    int_cm = []

    new_frame = pd.DataFrame(columns=["Image", "Mask", "Label", "Intensity"])
    
    for ii in range(total_test):
        if ii % args.patchn == 0:
            new_file_name = frame_test.loc[ii, "Image"].split("_patch")[0] + ".tif"
            new_file_name_mask = frame_test.loc[ii, "Mask"].split("_patch")[0] + "_Mask.tif"
            label = frame_test.loc[ii, "Label"]
            intensity = frame_test.loc[ii, "Intensity"]
            new_frame = new_frame.append({'Image': new_file_name, 'Mask': new_file_name_mask, 'Label': label, 'Intensity': intensity}, ignore_index=True)

    total_whole = len(new_frame)
    root_dir_whole = '/mnt/sdc1/upetruzzello/data/HEp-2_train_and_test/3_ICPR2014_Specimen_Dataset/datasets/train_ICPR2014_task2/train/'
    
    root_dir_2_whole = 'produced_images/produced_whole_mask_cross_val/'
    root_dir_fin_whole = 'produced_images/compared_whole_mask_cross_val/'
    result = np.zeros((1040, 1388, 3), np.uint8)
    TOTAL_DICE_WHOLE = 0
    dice = 0
    TOTAL_ACC_WHOLE = 0
    acc = 0
    
    c_h = 0
    c_s = 0
    c_n = 0
    c_c = 0
    c_nm = 0
    c_g = 0
    c_ms = 0
    dice_h = 0
    acc_h = 0
    dice_s = 0
    acc_s = 0
    dice_n = 0
    acc_n = 0
    dice_c = 0
    acc_c = 0
    dice_nm = 0
    acc_nm = 0
    dice_g = 0
    acc_g = 0
    dice_ms = 0
    acc_ms = 0
    
    for idx in range(total_whole):
        mask_name = os.path.join(root_dir_whole, new_frame.iloc[idx, 1])
        label_number = new_frame.iloc[idx, 2]
                
        if label_number == 0:
            label_to_check = 'homogeneous '
        elif label_number == 1:
            label_to_check = 'speckled '
        elif label_number == 2:
            label_to_check = 'nucleolar '
        elif label_number == 3:
            label_to_check = 'centromere '
        elif label_number == 4:
            label_to_check = 'golgi '
        elif label_number == 5:
            label_to_check = 'numem '
        else:
            label_to_check = 'mistp '
        
        mask_pred_name = os.path.join(root_dir_2_whole, new_frame.iloc[idx, 1])
        mask = cv2.imread(mask_name, cv2.IMREAD_GRAYSCALE)
        mask_pred = cv2.imread(mask_pred_name, cv2.IMREAD_GRAYSCALE)
        c = 0
        for ii in range(1040):
            for j in range(1388):
                if mask_pred[ii][j] == mask[ii][j]:
                    c += 1
                    if mask_pred[ii][j] == 0:
                        result[ii][j] = (0,0,0)
                    else:
                        result[ii][j] = (255,255,255)
                else:
                    if mask_pred[ii][j] == 0:
                        result[ii][j] = (0,0,255) #maschera = 1 predetta = 0 (False Negative) - Red
                    else:
                        result[ii][j] = (0,255,255) #maschera = 0 predetta = 1 (False Positive) - Yellow
        #calculate dice and segmentation accuracy
        calc_dice = dice_score(mask, mask_pred)
        calc_acc = c/(1040*1388)
        
        if label_to_check == 'homogeneous ':
            dice_h += calc_dice
            acc_h += calc_acc
            c_h += 1
        elif label_to_check == 'speckled ':
            dice_s += calc_dice
            acc_s += calc_acc
            c_s += 1
        elif label_to_check == 'nucleolar ':
            dice_n += calc_dice
            acc_n += calc_acc
            c_n += 1
        elif label_to_check == 'centromere ':
            dice_c += calc_dice
            acc_c += calc_acc
            c_c += 1
        elif label_to_check == 'numem ':
            dice_nm += calc_dice
            acc_nm += calc_acc
            c_nm += 1
        elif label_to_check == 'golgi ':
            dice_g += calc_dice
            acc_g += calc_acc
            c_g += 1
        else:
            dice_ms += calc_dice
            acc_ms += calc_acc
            c_ms += 1
        
        dice += calc_dice
        acc += calc_acc
        #save images
        fina_string = root_dir_fin_whole + new_frame.iloc[idx, 1]
        cv2.imwrite(fina_string, result)
    
    print("********************************************")
    print(c_h, c_s, c_n, c_c, c_nm, c_g, c_ms)
    
    if c_h != 0:
        t_dice_h = dice_h / c_h
        t_acc_h = acc_h / c_h
    else:
        t_dice_h = 0
        t_acc_h = 0
    
    if c_s != 0:
        t_dice_s = dice_s / c_s
        t_acc_s = acc_s / c_s
    else:
        t_dice_s = 0
        t_acc_s = 0
    
    if c_n != 0:
        t_dice_n = dice_n / c_n
        t_acc_n = acc_n / c_n
    else:
        t_dice_n = 0
        t_acc_n = 0
        
    if c_c != 0:
        t_dice_c = dice_c / c_c
        t_acc_c = acc_c / c_c
    else:
        t_dice_c = 0
        t_acc_c = 0
        
    if c_nm != 0:
        t_dice_nm = dice_nm / c_nm
        t_acc_nm = acc_nm / c_nm
    else:
        t_dice_nm = 0
        t_acc_nm = 0
        
    if c_g != 0:
        t_dice_g = dice_g / c_g
        t_acc_g = acc_g / c_g
    else:
        t_dice_g = 0
        t_acc_g = 0
        
    if c_ms != 0:
        t_dice_ms = dice_ms / c_ms
        t_acc_ms = acc_ms / c_ms
    else:
        t_dice_ms = 0
        t_acc_ms = 0   
    
    
    result_list_dice_acc = [t_dice_h, t_dice_s, t_dice_n, t_dice_c, t_dice_nm, t_dice_g, t_dice_ms, t_acc_h, t_acc_s, t_acc_n, t_acc_c, t_acc_nm, t_acc_g, t_acc_ms]
    
    with open('csv_result/result_whole_cross_val_shuff_dice_and_acc_per_class.csv', 'a', newline='') as f2:
        csv_writer = csv.writer(f2, delimiter=';')  
        csv_writer.writerow(result_list_dice_acc)
        
    result_list_dice_acc = []
    
    
    TOTAL_DICE_WHOLE = dice / total_whole
    TOTAL_ACC_WHOLE = acc / total_whole
    print("***********DICE SPECIMEN***********: ",TOTAL_DICE_WHOLE)
    print("***********SEG.ACC. SPECIMEN***********: ",TOTAL_ACC_WHOLE)
    
    #append result to csv
    result_list = [MCA_WHOLE, ACC_INT_WHOLE, TOTAL_DICE_WHOLE, TOTAL_ACC_WHOLE]
    
    with open('csv_result/result_whole_cross_val_shuff_5.csv', 'a', newline='') as f1:
        csv_writer = csv.writer(f1, delimiter=';')
        
        if i == 0:
            csv_writer.writerow(["Pattern Classification","Intensity Classification", "Dice Score", "Segmentation Accuracy"])
        
        csv_writer.writerow(result_list)
        
    result_list = []