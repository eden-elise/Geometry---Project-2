import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

########################################
# Example Data
########################################

vectors = [
    [0.2,0.5,0.3],
    [0.8,0.1],
    [0.4,0.7,0.2,0.9],
    [0.1,0.3,0.2],
    [0.9,0.8]
]

labels = [
    "English",
    "Arabic",
    "Chinese"
]

########################################
# Encode Languages
########################################

languages = sorted(list(set(labels)))
lang_to_id = {l:i for i,l in enumerate(languages)}
id_to_lang = {i:l for l,i in lang_to_id.items()}

y = [lang_to_id[l] for l in labels]

########################################
# Dataset
########################################

class LanguageDataset(Dataset):

    def __init__(self, vectors, labels):
        self.vectors = vectors
        self.labels = labels

    def __len__(self):
        return len(self.vectors)

    def __getitem__(self, idx):

        v = torch.tensor(self.vectors[idx], dtype=torch.float32).unsqueeze(1)
        label = torch.tensor(self.labels[idx])

        return v, label

dataset = LanguageDataset(vectors, y)

########################################
# Collate Function (for variable length)
########################################

def collate(batch):

    sequences = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch])

    lengths = torch.tensor([len(seq) for seq in sequences])

    padded = pad_sequence(sequences, batch_first=True)

    return padded, lengths, labels

loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate)

########################################
# Model
########################################

class LanguageLSTM(nn.Module):

    def __init__(self, hidden_size, num_languages):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, num_languages)

    def forward(self, x, lengths):

        packed = pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )

        _, (h_n, _) = self.lstm(packed)

        h = h_n[-1]

        return self.fc(h)

########################################
# Initialize
########################################

model = LanguageLSTM(hidden_size=32, num_languages=len(languages))

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

########################################
# Training
########################################

for epoch in range(200):

    for X, lengths, y in loader:

        pred = model(X, lengths)

        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if epoch % 20 == 0:
        print("epoch", epoch, "loss", loss.item())

########################################
# Prediction Example
########################################

test_vector = [0.3,0.6,0.2]

test = torch.tensor(test_vector).float().unsqueeze(1).unsqueeze(0)

length = torch.tensor([len(test_vector)])

logits = model(test, length)

prob = torch.softmax(logits, dim=1)

pred_id = torch.argmax(prob).item()

print("Probabilities:", prob.detach().numpy())
print("Predicted language:", id_to_lang[pred_id])