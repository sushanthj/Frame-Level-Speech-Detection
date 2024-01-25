import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchsummaryX import summary
from torch.utils.data import Dataset, DataLoader
import torchaudio
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable

import torchaudio.transforms as tat

from sklearn.metrics import accuracy_score
import gc

import zipfile
import pandas as pd
from tqdm import tqdm
import os
import datetime

# imports for decoding and distance calculation
import ctcdecode
import Levenshtein
from ctcdecode import CTCBeamDecoder

import warnings
warnings.filterwarnings('ignore')


device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
print("Device: ", device)



# ARPABET PHONEME MAPPING
# DO NOT CHANGE

CMUdict_ARPAbet = {
    "" : " ",
    "[SIL]": "-", "NG": "G", "F" : "f", "M" : "m", "AE": "@",
    "R"    : "r", "UW": "u", "N" : "n", "IY": "i", "AW": "W",
    "V"    : "v", "UH": "U", "OW": "o", "AA": "a", "ER": "R",
    "HH"   : "h", "Z" : "z", "K" : "k", "CH": "C", "W" : "w",
    "EY"   : "e", "ZH": "Z", "T" : "t", "EH": "E", "Y" : "y",
    "AH"   : "A", "B" : "b", "P" : "p", "TH": "T", "DH": "D",
    "AO"   : "c", "G" : "g", "L" : "l", "JH": "j", "OY": "O",
    "SH"   : "S", "D" : "d", "AY": "Y", "S" : "s", "IH": "I",
    "[SOS]": "[SOS]", "[EOS]": "[EOS]"
}

CMUdict = list(CMUdict_ARPAbet.keys())
ARPAbet = list(CMUdict_ARPAbet.values())


PHONEMES = CMUdict[:-2]

LABELS = ARPAbet[:-2]

root = "/home/mrsd_teamh/sush/11-785/11-785/Assignments/hw3/data/11-785-f23-hw3p2/"

class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, partition="train-clean-100", transforms=[]):
        # Load the directory and all files in them
        self.transforms = transforms
        self.mfcc_dir = os.path.join(root, partition, "mfcc")
        self.transcript_dir = os.path.join(root, partition, "transcript")
        mfcc_names = sorted(os.listdir(self.mfcc_dir))
        transcript_names = sorted(os.listdir(self.transcript_dir))

        assert len(mfcc_names) == len(transcript_names)

        self.mfccs = []
        self.transcripts = []

        # We need to do cepstral normalization for each mfcc
        # Define a cepstral mean and variance normalization transform
        # NOTE: Defualt for normalization is axis=0

        for i in range(len(mfcc_names)):
            # Load a single mfcc
            mfcc = np.load(os.path.join(self.mfcc_dir, mfcc_names[i]))
            # normalization
            mfcc = (mfcc - np.mean(mfcc, axis=0)) / (np.std(mfcc, axis=0) + 1e-8)
            self.mfccs.append(mfcc)
            transcript  = np.load(os.path.join(self.transcript_dir, mfcc_names[i]))
            self.transcripts.append(transcript[1:-1])

        # Final Dataset Attributes
        self.length = len(self.mfccs)
        # self.mfcc_files = np.concatenate(self.mfccs)
        self.mfcc_files = self.mfccs
        # self.transcript_files = np.concatenate(self.transcripts)

        self.PHONEMES = PHONEMES

        # HOW CAN WE REPRESENT PHONEMES? CAN WE CREATE A MAPPING FOR THEM?
        # HINT: TENSORS CANNOT STORE NON-NUMERICAL VALUES OR STRINGS
        phonemes_map = {phoneme: i for i, phoneme in enumerate(self.PHONEMES)}
        phonemes_lambda = lambda x: phonemes_map[x]
        # Convert the numpy array of strings (phonemes) to int
        self.transcript_files = [np.array(list(map(phonemes_lambda, x))) for x in self.transcripts]


    def __len__(self):
        return self.length

    def __getitem__(self, ind):
        mfcc = self.mfcc_files[ind]
        mfcc = torch.FloatTensor(mfcc)
        transcript = torch.tensor(self.transcript_files[ind])
        return (mfcc, transcript)


    def collate_fn(self,batch):
        # batch of input mfcc coefficients
        batch_mfcc = [item[0] for item in batch]
        # batch of output phonemes
        batch_transcript = [item[1] for item in batch]

        # HINT: CHECK OUT -> pad_sequence (imported above)
        # Also be sure to check the input format (batch_first)

        try:
          batch_mfcc_pad = pad_sequence(batch_mfcc, batch_first=True, padding_value=0)
          lengths_mfcc = torch.tensor([len(item) for item in batch_mfcc])
          # print("Shape of padded seq is", batch_mfcc_pad.shape)
        except:
          print("len of batch is ", len(batch_mfcc))
          for batch in batch_mfcc:
            print(batch.shape)

        batch_transcript_pad = pad_sequence(batch_transcript, batch_first=True, padding_value=0)
        lengths_transcript = torch.tensor([len(item) for item in batch_transcript])

        # perform transformation on whole batch instead of each input
        # torchaudio transforms expect input to be (batch, feature_vector, time) as the axes
        stacked_batch = batch_mfcc_pad
        stacked_batch = stacked_batch.permute(0, 2, 1)
        if len(self.transforms) != 0:
            for transform in self.transforms:
                batch_mfcc_pad = transform(stacked_batch)
        # unpermute to get back correct shape
        batch_mfcc_pad_transformed = stacked_batch.permute(0,2,1)

        # Return the following values: padded features, padded labels, actual length of features, actual length of the labels
        return batch_mfcc_pad_transformed, batch_transcript_pad, torch.tensor(lengths_mfcc), torch.tensor(lengths_transcript)


class AudioDatasetTest(torch.utils.data.Dataset):
   def __init__(self, partition="test-clean"):
      # Load the directory and all files in them
      self.mfcc_dir = os.path.join(root, partition, "mfcc")
      mfcc_names = sorted(os.listdir(self.mfcc_dir))

      self.mfccs = []

      # We need to do cepstral normalization for each mfcc
      # Define a cepstral mean and variance normalization transform

      for i in range(len(mfcc_names)):
          # Load a single mfcc
          mfcc = np.load(os.path.join(self.mfcc_dir, mfcc_names[i]))
          # normalization
          mfcc = (mfcc - np.mean(mfcc, axis=0)) / (np.std(mfcc, axis=0) + 1e-8)
          self.mfccs.append(mfcc)

      # Final Dataset Attributes
      self.length = len(self.mfccs)
      self.mfcc_files = self.mfccs


   def __len__(self):
      return self.length

   def __getitem__(self, ind):
      mfcc = self.mfcc_files[ind]
      mfcc = torch.FloatTensor(mfcc)
      return mfcc


   def collate_fn(self,batch):
      batch_mfcc = [mfcc for mfcc in batch]
      length_mfcc = [len(mfcc) for mfcc in batch]

      # HINT: CHECK OUT -> pad_sequence (imported above)
      # Also be sure to check the input format (batch_first)

      try:
        batch_mfcc_pad = pad_sequence(batch_mfcc, batch_first=True, padding_value=0)
        lengths_mfcc = torch.tensor(length_mfcc)
      except:
        for batch in batch_mfcc:
          print(batch.shape)

      # NOTE: the test dataset is for one sequence at a time, therefore, it lacks the second dimension
      # Let's introduce that dimension here
      # batch_mfcc_pad = batch_mfcc_pad.unsqueeze(1)

      # Return the following values: padded features, padded labels, actual length of features, actual length of the labels
      return batch_mfcc_pad, torch.tensor(lengths_mfcc)


   def collate_fn(self,batch):
      batch_mfcc = [mfcc for mfcc in batch]
      length_mfcc = [len(mfcc) for mfcc in batch]

      # HINT: CHECK OUT -> pad_sequence (imported above)
      # Also be sure to check the input format (batch_first)

      try:
        batch_mfcc_pad = pad_sequence(batch_mfcc, batch_first=True, padding_value=0)
        lengths_mfcc = torch.tensor(length_mfcc)
      except:
        for batch in batch_mfcc:
          print(batch.shape)

      # NOTE: the test dataset is for one sequence at a time, therefore, it lacks the second dimension
      # Let's introduce that dimension here
      # batch_mfcc_pad = batch_mfcc_pad.unsqueeze(1)

      # Return the following values: padded features, padded labels, actual length of features, actual length of the labels
      return batch_mfcc_pad, torch.tensor(lengths_mfcc)


# Feel free to add more items here
config = {
    "beam_width" : 2,
    "lr"         : 2e-3,
    "epochs"     : 100,
    "batch_size" : 64,  # Increase if your device can handle it
    'truncated_normal_mean' : 0,
    'truncated_normal_std' : 0.2,
}

# You may pass this as a parameter to the dataset class above
# This will help modularize your implementation
train_transforms = [torchaudio.transforms.FrequencyMasking(freq_mask_param=10), torchaudio.transforms.TimeMasking(time_mask_param=90)]
# train_transforms = []
valid_transforms = []

# get me RAMMM!!!!
import gc
gc.collect()

# Create objects for the dataset class
train_data = AudioDataset(partition="train-clean-100", transforms=train_transforms)
val_data = AudioDataset(partition="dev-clean", transforms=valid_transforms)
test_data = AudioDatasetTest(partition="test-clean")

# Do NOT forget to pass in the collate function as parameter while creating the dataloader
train_loader = torch.utils.data.DataLoader(
    dataset     = train_data,
    num_workers = 4,
    batch_size  = config['batch_size'],
    pin_memory  = True,
    shuffle     = True,
    collate_fn  = train_data.collate_fn
)

val_loader = torch.utils.data.DataLoader(
    dataset     = val_data,
    num_workers = 2,
    batch_size  = config['batch_size'],
    pin_memory  = True,
    shuffle     = False,
    collate_fn = val_data.collate_fn
)

test_loader = torch.utils.data.DataLoader(
    dataset     = test_data,
    num_workers = 2,
    batch_size  = config['batch_size'],
    pin_memory  = True,
    shuffle     = False,
    collate_fn  = test_data.collate_fn
)

print("Batch size: ", config['batch_size'])
print("Train dataset samples = {}, batches = {}".format(train_data.__len__(), len(train_loader)))
print("Val dataset samples = {}, batches = {}".format(val_data.__len__(), len(val_loader)))
print("Test dataset samples = {}, batches = {}".format(test_data.__len__(), len(test_loader)))

# sanity check
i = 0
for data in train_loader:
    x, y, lx, ly = data
    print(x.shape, y.shape, lx.shape, ly.shape)
    break

for data in val_loader:
    x, y, lx, ly = data
    print(x.shape, y.shape, lx.shape, ly.shape)
    break

for data in test_loader:
    x, lx = data
    print(x.shape, lx.shape)
    break


class CustomResidualBlock(torch.nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
      super(CustomResidualBlock, self).__init__()
      self.conv1 = nn.Conv1d(in_ch, 256, kernel_size=3,
                              stride=stride, padding=1, bias=False)
      self.bn1 = nn.BatchNorm1d(256)
      self.gelu = nn.GELU()
      self.conv2 = nn.Conv1d(256, out_ch, kernel_size=3,
                              stride=1, padding=1, bias=False)
      self.bn2 = nn.BatchNorm1d(out_ch)

      self.shortcut = nn.Sequential()
      if stride != 1 or in_ch != out_ch:
          self.shortcut = nn.Sequential(
              nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
              nn.BatchNorm1d(out_ch)
          )

    def forward(self, x):
      out = self.conv1(x)
      out = self.bn1(out)
      return out


# Utils for network
torch.cuda.empty_cache()

class PermuteBlock(torch.nn.Module):
    def forward(self, x):
        return x.transpose(1, 2)

class pBLSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(pBLSTM, self).__init__()

        # TODO: Initialize a single layer bidirectional LSTM with the given input_size and hidden_size
        self.blstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size, num_layers=2, bidirectional=True, batch_first=True)

    def forward(self, x_packed): # x_packed is a PackedSequence

        x_padded, x_lengths = pad_packed_sequence(x_packed, batch_first=True)

        x_downsampled, x_lengths_downsampled = self.trunc_reshape(x_padded, x_lengths)

        x_packed_downsampled = pack_padded_sequence(x_downsampled, x_lengths_downsampled, batch_first=True, enforce_sorted=False)

        output, output_lens = self.blstm(x_packed_downsampled)

        return output


    def trunc_reshape(self, x, x_lens):
      # Check if you have an odd number of time steps
      if x.size(1) % 2 != 0:
          # Remove the last time step
          x = x[:, :-1, :]
          # Update the lengths array to reflect the change
          x_lens = [length - 1 for length in x_lens]

      # Reshape x to reduce the number of time steps by a downsampling factor while increasing the number of features
      x_reshaped = x.view(x.size(0), x.size(1) // 2, x.size(2) * 2)
      # Reduce lengths by the same downsampling factor
      x_lens_shortened = [length // 2 for length in x_lens]

      return x_reshaped, x_lens_shortened


class LockedDropout(nn.Module):
    def __init__(self):
        super().__init__()

        self.dropout = 0.4

    # assuming input is packed_seq
    def forward(self, x_packed):
        if not self.training or self.dropout==0.0:
            return x_packed

        x, x_lens = pad_packed_sequence(x_packed, batch_first=True)
        m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - self.dropout)
        mask = Variable(m, requires_grad=False) / (1 - self.dropout)
        mask = mask.expand_as(x)
        x = mask * x

        x_packed = pack_padded_sequence(x, x_lens, batch_first=True, enforce_sorted=False)

        return x_packed


class Encoder(torch.nn.Module):
    '''
    The Encoder takes utterances as inputs and returns latent feature representations
    '''
    def __init__(self, input_size, encoder_hidden_size):
        super(Encoder, self).__init__()

        self.stride = 1

        self.embedding = CustomResidualBlock(in_ch=input_size, out_ch=encoder_hidden_size, stride=self.stride)
        # self.embedding = torch.nn.Conv1d(in_channels=input_size, out_channels=encoder_hidden_size, kernel_size=3, stride=1, padding=1)
        self.locked_dropout = LockedDropout()
        self.permute = PermuteBlock()

        self.pBLSTMs = torch.nn.Sequential(
            pBLSTM(input_size=encoder_hidden_size*2, hidden_size=encoder_hidden_size),
            LockedDropout(),
            pBLSTM(input_size=encoder_hidden_size*4, hidden_size=encoder_hidden_size),
            LockedDropout(),
        )

        for layer in self.pBLSTMs.children():
            if isinstance(layer, pBLSTM):
                # Initialize the weights of the pBLSTM layer
                for name, param in layer.named_parameters():
                    if 'weight' in name:
                        torch.nn.init.xavier_normal(param)

    def forward(self, x, x_lens):
        x = self.permute(x)
        x = self.embedding(x)
        x = self.permute(x)
        x_packed = pack_padded_sequence(x, x_lens, batch_first=True, enforce_sorted=False)
        encoder_outputs = self.pBLSTMs(x_packed)
        encoder_outputs, encoder_lens = pad_packed_sequence(encoder_outputs, batch_first=True)

        return encoder_outputs, encoder_lens


class Decoder(torch.nn.Module):

    def __init__(self, embed_size, output_size= 41):
        super().__init__()

        self.mlp = torch.nn.Sequential(
            PermuteBlock(), torch.nn.BatchNorm1d(embed_size), PermuteBlock(),
            torch.nn.Linear(embed_size, 1200),
            PermuteBlock(), torch.nn.BatchNorm1d(1200), PermuteBlock(),
            torch.nn.SiLU(),
            torch.nn.Linear(1200, 1200),
            torch.nn.SiLU(),
            torch.nn.Dropout(p=0.4),
            torch.nn.Linear(1200, output_size),
        )

        self.softmax = torch.nn.LogSoftmax(dim=2)

    def forward(self, encoder_out):
        mlp_out = self.mlp(encoder_out)

        out = self.softmax(mlp_out) # classification layer

        return out


class ASRModel(torch.nn.Module):

    def __init__(self, input_size=28, embed_size=256, output_size= len(PHONEMES)):
        super().__init__()

        self.encoder = Encoder(input_size, embed_size)
        self.decoder = Decoder(embed_size*2, output_size)

    def forward(self, x, lengths_x):

        encoder_out, encoder_lens = self.encoder(x, lengths_x)
        decoder_out = self.decoder(encoder_out)

        return decoder_out, encoder_lens


for data in val_loader:
    x, y, lx, ly = data
    print(x.shape, y.shape, lx.shape, ly.shape)
    break

model = ASRModel(
    output_size = len(PHONEMES)
).to(device)
print(model)
# summary(model, x.to(device), lx)

############ Criterion ##########################################################

criterion = torch.nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
# CTC Loss: https://pytorch.org/docs/stable/generated/torch.nn.CTCLoss.html

optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], betas=(0.9, 0.999), weight_decay=0.05)

# CTC Beam Decoder Doc: https://github.com/parlance/ctcdecode
decoder = CTCBeamDecoder(
    LABELS,
    beam_width=config["beam_width"],
    blank_id=0,
    log_probs_input=True
)

# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.7, patience=3)
gamma = 0.8
milestones = [20, 50, 70, 75, 80, 85, 90, 95]

# scheduler1 = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=0.9, total_iters=5)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

# Mixed Precision, if you need it
scaler = torch.cuda.amp.GradScaler()

############# Decoding Functions ####################################################

def decode_prediction(output, output_lens, decoder, PHONEME_MAP=LABELS):

    decoded_output, beam_scores, timesteps, out_lens = decoder.decode(output, seq_lens=output_lens) #lengths - list of lengths
    pred_strings = []

    # print("out_lens shape is ", out_lens.shape[0]) = batch_size
    for i in range(out_lens.shape[0]):
        best_string = decoded_output[i,0, 0:out_lens[i,0]]
        pred_string = ''.join([PHONEME_MAP[c] for c in best_string])
        pred_strings.append(pred_string)

    return pred_strings

def calculate_levenshtein(output, label, output_lens, label_lens, decoder, PHONEME_MAP=LABELS): # y - sequence of integers

    dist = 0
    batch_size = label.shape[0]

    pred_strings = decode_prediction(output, output_lens, decoder, PHONEME_MAP)

    for i in range(batch_size):
        pred_string = pred_strings[i]
        label_string = ''.join(PHONEME_MAP[p] for p in label[i, :label_lens[i]])
        dist += Levenshtein.distance(pred_string, label_string)

    dist /= batch_size
    return dist


############### Wandb Logging ############################################

import wandb
wandb.login(key="") #API Key is in your wandb account, under settings (wandb.ai/settings)

run = wandb.init(
    name = "Trial_12", ## Wandb creates random run names if you skip this field
    reinit = True, ### Allows reinitalizing runs when you re-run this cell
    # run_id = ### Insert specific run id here if you want to resume a previous run
    # resume = "must" ### You need this to resume previous runs, but comment out reinit = True when using this
    project = "hw3p2-ablations", ### Project should be created in your wandb account
    config = config ### Wandb Config for your run
)

############################ Train Functions ##############################

from tqdm import tqdm

def train_model(model, train_loader, criterion, optimizer):

    model.train()
    batch_bar = tqdm(total=len(train_loader), dynamic_ncols=True, leave=False, position=0, desc='Train')

    total_loss = 0

    for i, data in enumerate(train_loader):
        optimizer.zero_grad()

        x, y, lx, ly = data
        x, y = x.to(device), y.to(device)

        with torch.cuda.amp.autocast():
            h, lh = model(x, lx)
            h = torch.permute(h, (1, 0, 2))
            loss = criterion(h, y, lh, ly)

        total_loss += loss.item()

        batch_bar.set_postfix(
            loss="{:.04f}".format(float(total_loss / (i + 1))),
            lr="{:.06f}".format(float(optimizer.param_groups[0]['lr'])))

        batch_bar.update() # Update tqdm bar

        # Another couple things you need for FP16.
        scaler.scale(loss).backward() # This is a replacement for loss.backward()
        scaler.step(optimizer) # This is a replacement for optimizer.step()
        scaler.update() # This is something added just for FP16

        del x, y, lx, ly, h, lh, loss
        torch.cuda.empty_cache()

    batch_bar.close() # You need this to close the tqdm bar

    return total_loss / len(train_loader)


def validate_model(model, val_loader, decoder, phoneme_map=LABELS):

    model.eval()
    batch_bar = tqdm(total=len(val_loader), dynamic_ncols=True, position=0, leave=False, desc='Val')

    total_loss = 0
    vdist = 0

    for i, data in enumerate(val_loader):

        x, y, lx, ly = data
        x, y = x.to(device), y.to(device)

        with torch.inference_mode():
            h, lh = model(x, lx)
            h = torch.permute(h, (1, 0, 2))
            loss = criterion(h, y, lh, ly)

        total_loss += float(loss)
        vdist += calculate_levenshtein(torch.permute(h, (1, 0, 2)), y, lh, ly, decoder, phoneme_map)

        batch_bar.set_postfix(loss="{:.04f}".format(float(total_loss / (i + 1))), dist="{:.04f}".format(float(vdist / (i + 1))))

        batch_bar.update()

        del x, y, lx, ly, h, lh, loss
        torch.cuda.empty_cache()

    batch_bar.close()
    total_loss = total_loss/len(val_loader)
    val_dist = vdist/len(val_loader)
    return total_loss, val_dist

############################## Checkpoint Path ########################################

checkpoint_path = "/home/mrsd_teamh/sush/11-785/11-785/Assignments/hw3/checkpoints/trial_12.pth"

############################### Load Checkpoints #######################################

# Check if the checkpoint file exists
if os.path.exists(checkpoint_path):
    # If the checkpoint file exists, load the checkpoint
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = 0
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # start_epoch = checkpoint['epoch'] + 1  # Start from the next epoch
    # best_val_dist = checkpoint['val_dist']  # Update the best accuracy
    # Load the checkpoint and update the scheduler state if it exists in the checkpoint
    # if 'scheduler_state_dict' in checkpoint:
        # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        # print("Loaded scheduler state from checkpoint.")
    # else:
        # print("No scheduler state found in checkpoint.")
    print("Loaded checkpoint from:", checkpoint_path)
else:
    # If the checkpoint file does not exist, start training from scratch
    start_epoch = 0
    print("No checkpoint found at:", checkpoint_path)
###########################################################

########### MAIN TRAIN LOOP ###############################

best_val_dist = 1000

for epoch in range(config['epochs']):

    curr_lr = float(optimizer.param_groups[0]['lr'])

    train_loss = train_model(model, train_loader, criterion, optimizer)

    print("\nEpoch {}/{}: \n Train Loss {:.04f}\t Learning Rate {:.04f}".format(
        epoch + 1,
        config['epochs'],
        train_loss,
        curr_lr))

    valid_loss, valid_dist  = validate_model(model, val_loader, decoder, LABELS)

    print("Val loss {:.04f}%\t Val dist {:.04f}".format(valid_loss, valid_dist))

    wandb.log({"train_loss":train_loss, 'validation_Loss':valid_loss,
               'validation_dist': valid_dist, "learning_Rate": curr_lr})

    # If you are using a scheduler in your train function within your iteration loop, you may want to log
    # your learning rate differently

    # #Save model in drive location if val_acc is better than best recorded val_acc
    if valid_dist <= best_val_dist:
      #path = os.path.join(root, model_directory, 'checkpoint' + '.pth')
      print("Saving model")
      # save locally
      torch.save({'model_state_dict':model.state_dict(),
                  'optimizer_state_dict':optimizer.state_dict(),
                  'scheduler_state_dict':scheduler.state_dict(),
                  'val_dist': valid_dist,
                  'epoch': epoch}, './checkpoint.pth')
      # save in drive as well
      torch.save({'model_state_dict':model.state_dict(),
                  'optimizer_state_dict':optimizer.state_dict(),
                  'scheduler_state_dict':scheduler.state_dict(),
                  'val_dist': valid_dist,
                  'epoch': epoch}, checkpoint_path)

      best_val_dist = valid_dist
      # save in wandb
      wandb.save('checkpoint.pth')
      # You may find it interesting to exlplore Wandb Artifcats to version your models

    #   scheduler.step(valid_dist)
    scheduler.step()

run.finish()


############# Load Previous Model #########################

# Check if the checkpoint file exists
if os.path.exists(checkpoint_path):
    # If the checkpoint file exists, load the checkpoint
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = 0
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1  # Start from the next epoch
    best_val_dist = checkpoint['val_dist']  # Update the best accuracy
    # Load the checkpoint and update the scheduler state if it exists in the checkpoint
    if 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print("Loaded scheduler state from checkpoint.")
    else:
        print("No scheduler state found in checkpoint.")
    print("Loaded checkpoint from:", checkpoint_path)
else:
    # If the checkpoint file does not exist, start training from scratch
    start_epoch = 0
    print("No checkpoint found at:", checkpoint_path)
###########################################################


#TODO: Make predictions

test_decoder = CTCBeamDecoder(
    LABELS,
    beam_width=50,
    blank_id=0,
    log_probs_input=True
)

results = []

model.eval()
print("Testing")
for data in tqdm(test_loader):

    x, lx   = data
    x       = x.to(device)

    with torch.no_grad():
        h, lh = model(x, lx)

    prediction_strings = decode_prediction(h, lh, test_decoder, LABELS)
    # Save the output in the results array.
    results.extend(prediction_strings)

    del x, lx, h, lh
    torch.cuda.empty_cache()


data_dir = f"{root}/test-clean/random_submission.csv"
df = pd.read_csv(data_dir)
df.label = results
df.to_csv('submission.csv', index = False)
