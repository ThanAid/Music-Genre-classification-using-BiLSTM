import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import librosa.display
import matplotlib.pyplot as plt
import copy
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from torch.utils.data import SubsetRandomSampler, DataLoader
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.metrics import classification_report

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Input data files are available in the read-only "../input/" directory on kaggle:

####################### Important ###########################################
# Files must be in a file called "Data" in the same directory as this script
#############################################################################
for dirname, _, filenames in os.walk('Data'):
    for filename in filenames[:1]:
        print(os.path.join(dirname, filename))

spec = np.load('Data/fma_genre_spectrograms/train/80238.fused.full.npy')
spec2 = np.load('Data/fma_genre_spectrograms/train/69593.fused.full.npy')

# To decompose into the mel spectrogram and chromagram you can run:
mel, chroma = spec[:128], spec[128:]
mel2, chroma2 = spec2[:128], spec2[128:]
print(mel.shape, chroma.shape)

# Plot the spectrogram and the chromagram
fig, ax = plt.subplots(2, 2, figsize=(16, 12))
img = librosa.display.specshow(mel, x_axis='time', y_axis='linear', ax=ax[0, 0])
ax[0, 0].set(title='Spectrogram - Blues')
fig.colorbar(img, ax=ax[0, 0], format="%+2.f dB")

img = librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', ax=ax[1, 0])
ax[1, 0].set(title='Chromagram - Blues')
fig.colorbar(img, ax=ax[1, 0])

img = librosa.display.specshow(mel2, x_axis='time', y_axis='linear', ax=ax[0, 1])
ax[0, 1].set(title='Spectrogram - Trip-Hop')
fig.colorbar(img, ax=ax[0, 1], format="%+2.f dB")

img = librosa.display.specshow(chroma2, y_axis='chroma', x_axis='time', ax=ax[1, 1])
ax[1, 1].set(title='Chromagram - Trip-Hop')
fig.colorbar(img, ax=ax[1, 1])
fig.suptitle('Full Data', fontsize=16)
plt.show()

# Load beat sync data
spec_b = np.load('Data/fma_genre_spectrograms_beat/train/80238.fused.full.npy')
spec_b2 = np.load('Data/fma_genre_spectrograms_beat/train/69593.fused.full.npy')
mel_b, chroma_b = spec_b[:128], spec_b[128:]
mel_b2, chroma_b2 = spec_b2[:128], spec_b2[128:]

# Plot the spectrogram and the chromagram for the beat-synced

fig, ax = plt.subplots(2, 2, figsize=(16, 12))
img = librosa.display.specshow(mel_b, x_axis='time', y_axis='linear', ax=ax[0, 0])
ax[0, 0].set(title='Spectrogram - Blues')
fig.colorbar(img, ax=ax[0, 0], format="%+2.f dB")

img = librosa.display.specshow(chroma_b, y_axis='chroma', x_axis='time', ax=ax[1, 0])
ax[1, 0].set(title='Chromagram - Blues')
fig.colorbar(img, ax=ax[1, 0])

img = librosa.display.specshow(mel_b2, x_axis='time', y_axis='linear', ax=ax[0, 1])
ax[0, 1].set(title='Spectrogram - Trip-Hop')
fig.colorbar(img, ax=ax[0, 1], format="%+2.f dB")

img = librosa.display.specshow(chroma_b2, y_axis='chroma', x_axis='time', ax=ax[1, 1])
ax[1, 1].set(title='Chromagram - Trip-Hop')
fig.colorbar(img, ax=ax[1, 1])
fig.suptitle('Beat Sync Data', fontsize=16)
plt.show()

################################################################################
# Creating a Pytorch Dataset

# Combine similar classes and remove underrepresented classes
class_mapping = {
    'Rock': 'Rock',
    'Psych-Rock': 'Rock',
    'Indie-Rock': None,
    'Post-Rock': 'Rock',
    'Psych-Folk': 'Folk',
    'Folk': 'Folk',
    'Metal': 'Metal',
    'Punk': 'Metal',
    'Post-Punk': None,
    'Trip-Hop': 'Trip-Hop',
    'Pop': 'Pop',
    'Electronic': 'Electronic',
    'Hip-Hop': 'Hip-Hop',
    'Classical': 'Classical',
    'Blues': 'Blues',
    'Chiptune': 'Electronic',
    'Jazz': 'Jazz',
    'Soundtrack': None,
    'International': None,
    'Old-Time': None
}


# Helper functions to read fused, mel, and chromagram
def read_fused_spectrogram(spectrogram_file):
    spectrogram = np.load(spectrogram_file)
    return spectrogram.T


def read_mel_spectrogram(spectrogram_file):
    spectrogram = np.load(spectrogram_file)[:128]
    return spectrogram.T


def read_chromagram(spectrogram_file):
    spectrogram = np.load(spectrogram_file)[128:]
    return spectrogram.T


# Mapping the data
labels = pd.read_csv('Data/fma_genre_spectrograms/train_labels.txt', delim_whitespace=True, index_col=False)
labels_m = labels.copy()
labels_m["Genre"] = labels_m["Genre"].map(class_mapping)

# get the count of each genre
fig, ax = plt.subplots(1, 2, figsize=(16, 8))
labels['Genre'].value_counts().plot.barh(ax=ax[0])
ax[0].set(title='Histogram of Music Genres - Before mapping')
ax[0].set(xlabel='Genre')

labels_m['Genre'].value_counts().plot.barh(ax=ax[1])
ax[1].set(title='Histogram of Music Genres - After mapping')
ax[1].set(xlabel='Genre')
plt.show()


#######################################################################################################################

# It's useful to set the seed when debugging but when experimenting ALWAYS set seed=None to test the model in all cases.
def torch_train_val_split(
        dataset, batch_train, batch_eval,
        val_size=.2, shuffle=True, seed=None, debugging=False):
    # if debugging is True then we only get half of the dataset (to speed up)
    if debugging:
        evens = list(range(0, len(dataset), 2))
        dataset = torch.utils.data.Subset(dataset, evens)

    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    val_split = int(np.floor(val_size * dataset_size))
    if shuffle:
        # If shuffle is true we get random samples based on the indices
        np.random.seed(seed)
        np.random.shuffle(indices)
    train_indices = indices[val_split:]
    val_indices = indices[:val_split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(dataset,
                              batch_size=batch_train,
                              sampler=train_sampler)
    val_loader = DataLoader(dataset,
                            batch_size=batch_eval,
                            sampler=val_sampler)
    return train_loader, val_loader


class LabelTransformer(LabelEncoder):
    def inverse(self, y):
        try:
            return super(LabelTransformer, self).inverse_transform(y)
        except:
            return super(LabelTransformer, self).inverse_transform([y])

    def transform(self, y):
        try:
            return super(LabelTransformer, self).transform(y)
        except:
            return super(LabelTransformer, self).transform([y])


# Padding is needed because no all samples have the same dimensions
class PaddingTransform(object):
    def __init__(self, max_length, padding_value=0):
        self.max_length = max_length
        self.padding_value = padding_value

    def __call__(self, s):
        if len(s) == self.max_length:
            return s

        if len(s) > self.max_length:
            return s[:self.max_length]

        if len(s) < self.max_length:
            s1 = copy.deepcopy(s)
            pad = np.zeros((self.max_length - s.shape[0], s.shape[1]), dtype=np.float32)
            s1 = np.vstack((s1, pad))
            return s1


# Pytorch Dataset Class for creating the dataset
class SpectrogramDataset(Dataset):
    def __init__(self, path, class_mapping=None, train=True, max_length=-1, read_spec_fn=read_fused_spectrogram):
        t = 'train' if train else 'test'
        p = os.path.join(path, t)
        self.index = os.path.join(path, "{}_labels.txt".format(t))
        self.files, labels = self.get_files_labels(self.index, class_mapping)
        self.feats = [read_spec_fn(os.path.join(p, f)) for f in self.files]
        self.feat_dim = self.feats[0].shape[1]
        self.lengths = [len(i) for i in self.feats]
        self.max_length = max(self.lengths) if max_length <= 0 else max_length
        self.zero_pad_and_stack = PaddingTransform(self.max_length)
        self.label_transformer = LabelTransformer()
        if isinstance(labels, (list, tuple)):
            self.labels = np.array(self.label_transformer.fit_transform(labels)).astype('int64')

    def get_files_labels(self, txt, class_mapping):
        with open(txt, 'r') as fd:
            lines = [l.rstrip().split('\t') for l in fd.readlines()[1:]]
        files, labels = [], []
        for l in lines:
            label = l[1]
            if class_mapping:
                label = class_mapping[l[1]]
            if not label:
                continue
            # Kaggle automatically unzips the npy.gz format so this hack is needed
            _id = l[0].split('.')[0]
            npy_file = '{}.fused.full.npy'.format(_id)
            files.append(npy_file)
            labels.append(label)
        return files, labels

    def __getitem__(self, item):
        # output has samples|labels|lengths
        l = min(self.lengths[item], self.max_length)
        return self.zero_pad_and_stack(self.feats[item]), self.labels[item], l

    def __len__(self):
        return len(self.labels)


# Early stopping
''' This code is from https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py'''


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


############################################ LSTM  MODEL ##############################################################

class PadPackedSequence(nn.Module):
    def __init__(self):
        """Wrap sequence padding in nn.Module
        Args:
            batch_first (bool, optional): Use batch first representation. Defaults to True.
        """
        super(PadPackedSequence, self).__init__()
        self.batch_first = True
        self.max_length = None

    def forward(self, x):
        """Convert packed sequence to padded sequence
        Args:
            x (torch.nn.utils.rnn.PackedSequence): Packed sequence
        Returns:
            torch.Tensor: Padded sequence
        """
        out, lengths = pad_packed_sequence(
            x, batch_first=self.batch_first, total_length=self.max_length  # type: ignore
        )
        lengths = lengths.to(out.device)
        return out, lengths  # type: ignore


class PackSequence(nn.Module):
    def __init__(self):
        """Wrap sequence packing in nn.Module
        Args:
            batch_first (bool, optional): Use batch first representation. Defaults to True.
        """
        super(PackSequence, self).__init__()
        self.batch_first = True

    def forward(self, x, lengths):
        """Pack a padded sequence and sort lengths
        Args:
            x (torch.Tensor): Padded tensor
            lengths (torch.Tensor): Original lengths befor padding
        Returns:
            Tuple[torch.nn.utils.rnn.PackedSequence, torch.Tensor]: (packed sequence, sorted lengths)
        """
        lengths = lengths.to("cpu")
        out = pack_padded_sequence(
            x, lengths, batch_first=self.batch_first, enforce_sorted=False
        )

        return out


class LSTMBackbone(nn.Module):
    def __init__(
            self,
            input_dim,
            output_dim,
            rnn_size=128,
            num_layers=1,
            bidirectional=False,
            dropout=0.1,
    ):
        super(LSTMBackbone, self).__init__()
        self.batch_first = True
        self.bidirectional = bidirectional
        self.feature_size = rnn_size * 2 if self.bidirectional else rnn_size

        self.input_dim = input_dim
        self.rnn_size = rnn_size
        self.num_layers = num_layers
        self.hidden_size = rnn_size
        self.pack = PackSequence()
        self.unpack = PadPackedSequence()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=rnn_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=self.bidirectional,
            dropout=dropout,
        )
        self.drop = nn.Dropout(dropout)
        self.linear = nn.Linear(self.feature_size, output_dim)

    def forward(self, x, lengths):
        """LSTM forward
        Args:
            x (torch.Tensor):
                [B, S, F] Batch size x sequence length x feature size
                padded inputs
            lengths (torch.tensor):
                [B] Original lengths of each padded sequence in the batch
        Returns:
            torch.Tensor:
                [B, H] Batch size x hidden size lstm last timestep outputs
                2 x hidden_size if bidirectional
        """
        packed = self.pack(x, lengths)
        output, _ = self.lstm(packed)
        output, lengths = self.unpack(output)
        output = self.drop(output)

        rnn_all_outputs, last_timestep = self._final_output(output, lengths)
        # Use the last_timestep for classification / regression
        last_timestep = self.linear(last_timestep)

        # Alternatively rnn_all_outputs can be used with an attention mechanism
        return last_timestep

    def _merge_bi(self, forward, backward):
        """Merge forward and backward states
        Args:
            forward (torch.Tensor): [B, L, H] Forward states
            backward (torch.Tensor): [B, L, H] Backward states
        Returns:
            torch.Tensor: [B, L, 2*H] Merged forward and backward states
        """
        return torch.cat((forward, backward), dim=-1)

    def _final_output(self, out, lengths):
        """Create RNN ouputs
        Collect last hidden state for forward and backward states
        Code adapted from https://stackoverflow.com/a/50950188
        Args:
            out (torch.Tensor): [B, L, num_directions * H] RNN outputs
            lengths (torch.Tensor): [B] Original sequence lengths
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (
                merged forward and backward states [B, L, H] or [B, L, 2*H],
                merged last forward and backward state [B, H] or [B, 2*H]
            )
        """

        if not self.bidirectional:
            return out, self._select_last_unpadded(out, lengths)

        forward, backward = (out[..., : self.hidden_size], out[..., self.hidden_size:])
        # Last backward corresponds to first token
        last_backward_out = backward[:, 0, :] if self.batch_first else backward[0, ...]
        # Last forward for real length or seq (unpadded tokens)
        last_forward_out = self._select_last_unpadded(forward, lengths)
        out = self._merge_bi(forward, backward)

        return out, self._merge_bi(last_forward_out, last_backward_out)

    def _select_last_unpadded(self, out, lengths):
        """Get the last timestep before padding starts
        Args:
            out (torch.Tensor): [B, L, H] Fprward states
            lengths (torch.Tensor): [B] Original sequence lengths
        Returns:
            torch.Tensor: [B, H] Features for last sequence timestep
        """
        gather_dim = 1  # Batch first
        gather_idx = (
            (lengths - 1)  # -1 to convert to indices
            .unsqueeze(1)  # (B) -> (B, 1)
            .expand((-1, self.hidden_size))  # (B, 1) -> (B, H)
            # (B, 1, H) if batch_first else (1, B, H)
            .unsqueeze(gather_dim)
        )
        # Last forward for real length or seq (unpadded tokens)
        last_out = out.gather(gather_dim, gather_idx).squeeze(gather_dim)

        return last_out


############################################ TRAIN FUCNTION ############################################################

def train_model_lstm(model, train_loader, optimizer, criterion, n_epochs=100
                     , val=None, patience=None, overfit_batch=False, device=device, ver=1):
    ''' Train the model
    '''
    if overfit_batch:
        # if overfit batch is true then model will be trained using only 1 batch
        model.train()

        loss_train = []

        for x_batch, y_batch, lengths_train in train_loader:
            # Move batches to device
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            for epoch in range(1000):  # to train for a lot of epochs
                loss_epoch = []
                # Clear gradients
                optimizer.zero_grad()

                prediction = model(x_batch, lengths_train)

                # Forward pass
                loss = criterion(prediction, y_batch)

                # Backward and optimize
                loss.backward()
                # Update parameters
                optimizer.step()

                loss_epoch.append(loss.data.item())
                loss_train.append(np.mean(loss_epoch))

                if epoch % 10 == 0:
                    print(f'Epoch [{epoch + 1}/{1000}], Loss: {np.mean(loss_epoch):.4f}')

            break
        return model, loss_train

    # if overfit batch is false then train normally
    else:
        # Check if early stop is enabled:
        if patience is not None:
            # Initialize EarlyStopping
            early_stopping = EarlyStopping(patience=patience)

        loss_train = []  # store the mean loss for each epoch of training
        loss_val = []  # store the mean loss for each epoch of validation
        for epoch in range(n_epochs):
            model.train()
            loss_epoch = []

            for x_batch, y_batch, lengths_train in train_loader:
                # Move batches to device
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)

                # Clear gradients
                optimizer.zero_grad()

                prediction = model(x_batch, lengths_train)

                # Forward pass
                loss = criterion(prediction, y_batch)

                # Backward and optimize
                loss.backward()
                # Update parameters
                optimizer.step()

                loss_epoch.append(loss.data.item())

            loss_train.append(np.mean(loss_epoch))

            if epoch % ver == 0:
                print(f'Epoch [{epoch + 1}/{n_epochs}], Loss: {np.mean(loss_epoch):.4f}')

            if val is not None:
                with torch.no_grad():  # no gradients required!! eval mode, speeds up computation
                    loss_val_epoch = []
                    # turn off the regularisation during evaluation
                    model.eval()
                    for X_val, y_val, lengths_val in val:
                        X_val, y_val = X_val.to(device), y_val.to(device)

                        preds = model(X_val, lengths_val)  # get net's predictions
                        loss = criterion(preds, y_val)

                        loss_val_epoch.append(loss.data.item())

                    loss_val.append(np.mean(loss_val_epoch))

                if epoch % ver == 0:
                    print(f'Epoch [{epoch + 1}/{n_epochs}], Loss (on validation): {np.mean(loss_val_epoch):.4f}')

                if patience is not None:
                    # Check if our patience is over (validation loss increased for given steps)
                    early_stopping(np.array(np.mean(loss_val_epoch)), model)

                    if early_stopping.early_stop:
                        print('Out of Patience. Early stopping... ')
                        break

            # checks if we will go back to the checkpoint
        if patience != -1 and early_stopping.early_stop == True:
            print('Loading model from checkpoint...')
            model.load_state_dict(torch.load('checkpoint.pt'))
            print('Checkpoint loaded.')

        # visualize the loss as the network trained
        fig = plt.figure(figsize=(10, 8))
        plt.plot(range(1, len(loss_train) + 1), loss_train, label='Training Loss')
        plt.plot(range(1, len(loss_val) + 1), loss_val, label='Validation Loss')

        # find position of lowest validation loss
        minposs = loss_val.index(min(loss_val)) + 1
        plt.axvline(minposs, linestyle='--', color='r', label='Early Stopping Checkpoint')

        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.ylim(0, max(loss_val + loss_train))  # consistent scale
        plt.xlim(0, len(loss_train) + 1)  # consistent scale
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.title('Validation and Training Loss of BiLSTM')
        plt.show()

    return model


############################################# PREDICT FUNCTION ########################################################

def predict_model_lstm(model, test_loader, criteriion, device=device):
    # Make predictions using model
    preds = []
    true_values = []
    loss = 0
    model.eval()  # prep model for evaluation

    with torch.no_grad():
        for x_batch, y_batch, lengths in test_loader:
            # move to device
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            # Make predictions
            pred = model(x_batch, lengths)

            preds.append(np.argmax(pred.cpu().numpy(), axis=1)[0])
            true_values.append(y_batch.cpu().numpy()[0])
            loss += criteriion(pred, y_batch)

        # Calculate Accuracy
        accuracy = sum(np.array(preds) == np.array(true_values)) / len(true_values)

    return preds, true_values, accuracy


########################################################################################################################
########################################################################################################################
########################################################################################################################
###################################### Overfitting with single batch - beat sync ######################################

# Getting the data from the beat-synced data folder
specs = SpectrogramDataset(
    'Data/fma_genre_spectrograms_beat/',
    train=True,
    class_mapping=class_mapping,
    max_length=-1,
    read_spec_fn=read_mel_spectrogram)

# splitting the data and using Dataloader
train_loader, val_loader = torch_train_val_split(specs, 32, 32, val_size=.33)

# Initializing the model
lstm = LSTMBackbone(input_dim=128, output_dim=10, rnn_size=100, num_layers=2, bidirectional=True, dropout=0)
lstm.double()
lstm.to(device)

print('\n--------------------Training lstm----------------------')

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(lstm.parameters(), lr=0.0001, weight_decay=0)  # weight_decay>0 for L2 regularization

# Train the model
lstm, losses = train_model_lstm(lstm, train_loader, optimizer, criterion, n_epochs=100,
                                patience=1, val=val_loader,
                                overfit_batch=True)  # overfit_batch=True to train with a single batch

# visualize the loss as the network trained
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(losses) + 1), losses, label='Training Loss')

plt.xlabel('epochs')
plt.ylabel('loss')
plt.ylim(0, max(losses))  # consistent scale
plt.xlim(0, len(losses) + 1)  # consistent scale
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.title('Overfit by training with a single batch', fontsize=16)
plt.show()

########################################################################################################################
########################################################################################################################
########################################################################################################################
###################################### Training using beat-sync Spectograms ######################################
batch_size = 64  # setting the batch size found to be optimal

# Getting the data from the beat-synced data folder
specs = SpectrogramDataset(
    'Data/fma_genre_spectrograms_beat/',
    train=True,
    class_mapping=class_mapping,
    max_length=-1,
    read_spec_fn=read_mel_spectrogram)

train_loader, val_loader = torch_train_val_split(specs, batch_size, batch_size, val_size=.33, debugging=False)

test_set = SpectrogramDataset(
    'Data/fma_genre_spectrograms_beat/',
    train=False,
    class_mapping=class_mapping,
    max_length=-1,
    read_spec_fn=read_mel_spectrogram)

# using dataloader on test data
test_loader = DataLoader(test_set, batch_size=1)

# Initializing the model
lstm_b = LSTMBackbone(input_dim=128, output_dim=10, rnn_size=150, num_layers=3, bidirectional=True, dropout=0.2)
lstm_b.double()
lstm_b.to(device)

print('--------------------Training lstm using beat- synced spectograms----------------------')

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(lstm_b.parameters(), lr=0.00005, weight_decay=1e-4)  # weight_decay>0 for L2 regularization

# Train the model
lstm_b = train_model_lstm(lstm_b, train_loader, optimizer, criterion, n_epochs=500,
                          patience=15, val=val_loader, overfit_batch=False, ver=10)

# Get predictions
preds_b, true_val_b, accu_b = predict_model_lstm(lstm_b, test_loader, criteriion=criterion, device=device)

# print the classification report
print('\nClassificatior report for BiLSTM using beat-synced spectograms:')
print(classification_report(true_val_b, preds_b))

########################################################################################################################
########################################################################################################################
########################################################################################################################
###################################### Training using Spectograms ######################################
batch_size = 64  # setting the batch size found to be optimal

specs = SpectrogramDataset(
    'Data/fma_genre_spectrograms/',
    train=True,
    class_mapping=class_mapping,
    max_length=-1,
    read_spec_fn=read_mel_spectrogram)

train_loader, val_loader = torch_train_val_split(specs, batch_size, batch_size, val_size=.33, debugging=False)

test_set = SpectrogramDataset(
    'Data/fma_genre_spectrograms/',
    train=False,
    class_mapping=class_mapping,
    max_length=-1,
    read_spec_fn=read_mel_spectrogram)

test_loader = DataLoader(test_set, batch_size=1)

# initializing the model
lstm_s = LSTMBackbone(input_dim=128, output_dim=10, rnn_size=150, num_layers=3, bidirectional=True, dropout=0.2)
lstm_s.double()
lstm_s.to(device)

print('\n--------------------Training lstm using spectograms----------------------')

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(lstm_s.parameters(), lr=0.00005, weight_decay=1e-4)  # weight_decay>0 for L2 regularization

# Train the model
lstm_s = train_model_lstm(lstm_s, train_loader, optimizer, criterion, n_epochs=500,
                          patience=15, val=val_loader, overfit_batch=False, ver=10)

# Getting the predictions
preds_s, true_val_s, accu_s = predict_model_lstm(lstm_s, test_loader, criteriion=criterion, device=device)

print('\nClassificatior report for BiLSTM using spectograms:')
print(classification_report(true_val_s, preds_s))

########################################################################################################################
########################################################################################################################
########################################################################################################################
###################################### Training using beat-synced Chromograms ######################################
batch_size = 64  # setting the batch size found to be optimal

specs = SpectrogramDataset(
    'Data/fma_genre_spectrograms_beat/',
    train=True,
    class_mapping=class_mapping,
    max_length=-1,
    read_spec_fn=read_chromagram)

train_loader, val_loader = torch_train_val_split(specs, batch_size, batch_size, val_size=.33, debugging=False)

test_set = SpectrogramDataset(
    'Data/fma_genre_spectrograms_beat/',
    train=False,
    class_mapping=class_mapping,
    max_length=-1,
    read_spec_fn=read_chromagram)

test_loader = DataLoader(test_set, batch_size=1)

# initializing the model
lstm_c = LSTMBackbone(input_dim=12, output_dim=10, rnn_size=150, num_layers=3, bidirectional=True, dropout=0.2)
lstm_c.double()
lstm_c.to(device)

print('\n--------------------Training lstm using chromograms----------------------')

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(lstm_c.parameters(), lr=0.00005, weight_decay=1e-4)  # weight_decay>0 for L2 regularization

# Train the model
lstm_c = train_model_lstm(lstm_c, train_loader, optimizer, criterion, n_epochs=500,
                          patience=15, val=val_loader, overfit_batch=False, ver=10)

# get predictions
preds_c, true_val_c, accu_c = predict_model_lstm(lstm_c, test_loader, criteriion=criterion, device=device)

print('\nClassificatior report for BiLSTM using beat-synced chromograms:')
print(classification_report(true_val_c, preds_c))

########################################################################################################################
########################################################################################################################
########################################################################################################################
###################################### Training using beat-synced fused Chromograms + Spectograms #####################
batch_size = 64  # setting the batch size found to be optimal

specs = SpectrogramDataset(
    'Data/fma_genre_spectrograms_beat/',
    train=True,
    class_mapping=class_mapping,
    max_length=-1,
    read_spec_fn=read_fused_spectrogram)

train_loader, val_loader = torch_train_val_split(specs, batch_size, batch_size, val_size=.33, debugging=False)

test_set = SpectrogramDataset(
    'Data/fma_genre_spectrograms_beat/',
    train=False,
    class_mapping=class_mapping,
    max_length=-1,
    read_spec_fn=read_fused_spectrogram)

test_loader = DataLoader(test_set, batch_size=1)

# Initialising the model
lstm_f = LSTMBackbone(input_dim=140, output_dim=10, rnn_size=150, num_layers=3, bidirectional=True, dropout=0.2)
lstm_f.double()
lstm_f.to(device)

print('\n--------------------Training lstm using fused data----------------------')

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(lstm_f.parameters(), lr=0.00005, weight_decay=1e-4)  # weight_decay>0 for L2 regularization

# Train the model
lstm_f = train_model_lstm(lstm_f, train_loader, optimizer, criterion, n_epochs=500,
                          patience=15, val=val_loader, overfit_batch=False, ver=10)

# getting the predictions
preds_f, true_val_f, accu_f = predict_model_lstm(lstm_f, test_loader, criteriion=criterion, device=device)

print('\nClassificatior report for BiLSTM using fused beat-synced chromograms and Spectograms:')
print(classification_report(true_val_f, preds_f))