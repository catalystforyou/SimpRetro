import numpy as np
import os
from tqdm import tqdm
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from rdkit import Chem
from rdkit.Chem import AllChem
import pickle

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

if os.path.exists('fingerprint_base.pkl'):
    fingerprint_base = pickle.load(open('fingerprint_base.pkl', 'rb'))
else:
    fingerprint_base = {}

def smiles_to_fingerprint(smiles, fp_length=2048, radius=2):
    mol = Chem.MolFromSmiles(smiles)
    return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=fp_length)).reshape(1, -1)

def smarts_to_fingerprint(smarts):
    rxn = AllChem.ReactionFromSmarts(smarts)
    return np.concatenate([np.array(AllChem.CreateDifferenceFingerprintForReaction(rxn).ToList()).reshape(1, -1), np.array(AllChem.CreateStructuralFingerprintForReaction(rxn).ToList()).reshape(1, -1)], axis=1)

def fingerprint_collate_fn(batch):
    global fingerprint_base
    length = len(fingerprint_base)
    fps = []
    targets = []
    for (rxn, p, label) in batch:
        if rxn not in fingerprint_base:
            fingerprint_base[rxn] = smarts_to_fingerprint(rxn)
        if p not in fingerprint_base:
            fingerprint_base[p] = smiles_to_fingerprint(p)
        fps.append(np.concatenate([fingerprint_base[rxn], fingerprint_base[p]], axis=1).reshape(-1, 1).squeeze())
        targets.append(label)
    if len(fingerprint_base) > length:
        pickle.dump(fingerprint_base, open('fingerprint_base.pkl', 'wb'))
        print('added {} fingerprints to fingerprint_base.pkl'.format(len(fingerprint_base) - length))
    fps = torch.tensor(np.array(fps), dtype=torch.float32) # .to('cuda')
    targets = torch.tensor(targets, dtype=torch.float32) # .to('cuda')
    return fps, targets

class MoleculeDataset(Dataset):
    def __init__(self):
        self.raw = []
        global fingerprint_base
        length = len(fingerprint_base)
        for i in range(20):
            
            if os.path.exists('../../r-smiles/roundtrip/labeled_data_{}.json'.format(i)):
                print('Loading labeled_data_{}.json'.format(i))
                raw = json.load(open('../../r-smiles/roundtrip/labeled_data_{}.json'.format(i), 'r'))
                for r in raw:
                    rxn, p, label = r[1:]
                    if rxn not in fingerprint_base:
                        fingerprint_base[rxn] = smarts_to_fingerprint(rxn)
                    if p not in fingerprint_base:
                        fingerprint_base[p] = smiles_to_fingerprint(p)
                    self.raw.append((rxn, p, label))
        if len(fingerprint_base) > length:
            print('added {} fingerprints to fingerprint_base.pkl'.format(len(fingerprint_base) - length))
            pickle.dump(fingerprint_base, open('fingerprint_base.pkl', 'wb'))
    def __len__(self):
        return len(self.raw)

    def __getitem__(self, idx):
        return self.raw[idx]

class LabelSmoothingBCELoss(nn.Module):
    def __init__(self, label_smoothing=0.1):
        super(LabelSmoothingBCELoss, self).__init__()
        self.label_smoothing = label_smoothing

    def forward(self, input, target):
        target_smooth = target * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        loss = F.binary_cross_entropy(input, target_smooth, reduction='mean')
        return loss

class Net_orig(nn.Module):
    def __init__(self):
        super(Net_orig, self).__init__()
        self.fc1 = nn.Linear(2048*4, 2048)
        self.fc2 = nn.Linear(2048, 1)
        self.bn = nn.BatchNorm1d(2048)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.bn(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x


def train(epoch, threshold, model, criterion, optimizer, train_loader):
    train_acc, train_recall, train_precision, train_f1 = 0, 0, 0, 0
    recall_total, precision_total = 0, 0
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to('cuda'), target.to('cuda')
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output.squeeze(), target)
        train_acc += ((output.squeeze() > threshold) == target).sum().item()
        train_recall += ((output.squeeze() > threshold) & (target == 1)).sum().item()
        recall_total += (target == 1).sum().item()
        train_precision += ((output.squeeze() > threshold) & (target == 1)).sum().item()
        precision_total += (output.squeeze() > threshold).sum().item()
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    train_acc /= len(train_loader.dataset)
    train_recall /= recall_total
    if precision_total == 0:
        train_precision = 0
    else:
        train_precision /= precision_total
    if train_recall + train_precision == 0:
        train_f1 = 0
    else:
        train_f1 = 2 * train_recall * train_precision / (train_recall + train_precision)
    print(f'Train set: Accuracy: {train_acc:.4f}, Recall: {train_recall:.4f}, Precision: {train_precision:.4f}, F1: {train_f1:.4f}')

def test(loader, threshold, model, criterion):
    global best_f1
    model.eval()
    test_loss = 0
    test_acc, test_recall, test_precision, test_f1 = 0, 0, 0, 0
    recall_total, precision_total = 0, 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to('cuda'), target.to('cuda')
            output = model(data)
            test_loss += criterion(output.squeeze(), target).item()
            test_acc += ((output.squeeze() > threshold) == target).sum().item()
            test_recall += ((output.squeeze() > threshold) & (target == 1)).sum().item()
            recall_total += (target == 1).sum().item()
            test_precision += ((output.squeeze() > threshold) & (target == 1)).sum().item()
            precision_total += (output.squeeze() > threshold).sum().item()
    test_loss /= len(loader.dataset)
    test_acc /= len(loader.dataset)
    test_recall /= recall_total
    if precision_total == 0:
        test_precision = 0
    else:
        test_precision /= precision_total
    if test_recall + test_precision == 0:
        test_f1 = 0
    else:
        test_f1 = 2 * test_recall * test_precision / (test_recall + test_precision)
    if test_f1 > best_f1:
        best_f1 = test_f1
        torch.save(model.state_dict(), 'model_smoothbce.pth')

    
    print(f'\nTest set: Average loss: {test_loss:.4f}\n')
    print(f'Test set: Accuracy: {test_acc:.4f}, Recall: {test_recall:.4f}, Precision: {test_precision:.4f}, F1: {test_f1:.4f}\n')


def get_fps_for_temps(temps):
    fps = []
    length = len(fingerprint_base)
    for temp in temps:
        if temp not in fingerprint_base:
            fingerprint_base[temp] = smarts_to_fingerprint(temp)
        fps.append(fingerprint_base[temp])
    if len(fingerprint_base) > length:
        pickle.dump(fingerprint_base, open('fingerprint_base.pkl', 'wb'))
        print('added {} fingerprints to fingerprint_base.pkl'.format(len(fingerprint_base) - length))
    return np.array(fps)

def inference_all_temps(temps_fps, p, model):
    p_fp = smiles_to_fingerprint(p)
    model.eval()
    with torch.no_grad():
        data = torch.tensor(np.concatenate([temps_fps.squeeze(), np.repeat(p_fp, len(temps_fps), axis=0)], axis=1), dtype=torch.float32).to('cuda')
        output = model(data)
    return output.squeeze().cpu().numpy()

if __name__ == '__main__':
    
    
    model = Net_orig()
    model.to('cuda')
    dataset = MoleculeDataset()
    dataset_size = len(dataset)
    train_size = int(dataset_size * 0.8)
    val_size = int(dataset_size * 0.1)
    test_size = dataset_size - (train_size + val_size)
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=20480, shuffle=True, collate_fn=fingerprint_collate_fn, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=20480, shuffle=False, collate_fn=fingerprint_collate_fn, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=20480, shuffle=False, collate_fn=fingerprint_collate_fn, num_workers=4)
    
    criterion = LabelSmoothingBCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-5)
    threshold = 0.5
    best_f1 = 0
    np.random.seed(0)
    for epoch in range(100):
        train(epoch, model=model, criterion=criterion, optimizer=optimizer, train_loader=train_loader)
        test(val_loader, model=model, criterion=criterion)
    test(test_loader)