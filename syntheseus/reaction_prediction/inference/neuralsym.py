from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, List, Optional, Union

from rdkit import Chem

from syntheseus.interface.models import BackwardPredictionList, BackwardReactionModel, PredictionList
from syntheseus.interface.molecule import Molecule
from syntheseus.reaction_prediction.utils.inference import (
    get_module_path,
    get_unique_file_in_dir,
    process_raw_smiles_outputs,
)
from syntheseus.reaction_prediction.utils.misc import suppress_outputs
import torch
from collections import OrderedDict
from tqdm import tqdm
from torch import nn
from rdchiral.main import rdchiralRunText, rdchiralRun
import numpy as np
from rdkit.Chem import AllChem
from rdkit import Chem
from collections import defaultdict
from rdkit import RDLogger
import torch.nn.functional as F

class RolloutPolicyNet(nn.Module):
    def __init__(self, n_rules, fp_dim=2048, dim=512,
                 dropout_rate=0.3):
        super(RolloutPolicyNet, self).__init__()
        self.fp_dim = fp_dim
        self.n_rules = n_rules
        self.dropout_rate = dropout_rate
        self.fc1 = nn.Linear(fp_dim,dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        # self.fc2 = nn.Linear(dim,dim)
        # self.bn2 = nn.BatchNorm1d(dim)
        # self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(dim,n_rules)

    def forward(self,x, y=None, loss_fn =nn.CrossEntropyLoss()):
        x = self.dropout1(F.elu(self.bn1(self.fc1(x))))
        # x = self.dropout1(F.elu(self.fc1(x)))
        # x = self.dropout2(F.elu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        if y is not None :
            return loss_fn(x, y)
        else :
            return x
        return x
    
def merge(reactant_d):
    ret = []
    for reactant, l in reactant_d.items():
        ss, ts = zip(*l)
        ret.append((reactant, sum(ss), list(ts)[0]))
    reactants, scores, templates = zip(*sorted(ret,key=lambda item : item[1], reverse=True))
    return list(reactants), list(scores), list(templates)

def load_parallel_model(state_path, template_rule_path,fp_dim=2048):
    template_rules = {}
    with open(template_rule_path, 'r') as f:
        for i, l in tqdm(enumerate(f), desc='template rules'):
            rule= l.strip()
            template_rules[rule] = i
    idx2rule = {}
    for rule, idx in template_rules.items():
        idx2rule[idx] = rule
    rollout = RolloutPolicyNet(len(template_rules),fp_dim=fp_dim)
    checkpoint = torch.load(state_path,map_location='cpu')
    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        name = k[7:]
        new_state_dict[name] = v
    rollout.load_state_dict(new_state_dict)
    return rollout, idx2rule

def preprocess(X,fp_dim):

    # Compute fingerprint from mol to feature
    mol = Chem.MolFromSmiles(X)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=int(fp_dim),useChirality=True)
    onbits = list(fp.GetOnBits())
    arr = np.zeros(fp.GetNumBits())
    arr[onbits] = 1
    # arr = (arr - arr.mean())/(arr.std() + 0.000001)
    # arr = arr / fp_dim
    # X = fps_to_arr(X)
    return arr

class NeuralSymModel(BackwardReactionModel):
    def __init__(
        self,
        model_dir: Union[str, Path],  # = '/work02/home/jrli/syntheseus/syntheseus/reaction_prediction/environments/external/neuralsym/models',
        device: str = "cuda:0",
        fp_dim: int =2048):
        """Initializes the NeuralSym model wrapper."""
        self.fp_dim = fp_dim
        state_path = model_dir + '/saved_rollout_state_1_2048.ckpt'
        template_path = model_dir + '/template_rules_1.dat'
        print(state_path, template_path)
        self.net, self.idx2rules = load_parallel_model(state_path,template_path, fp_dim)
        self.net.eval()
        self.device = device
        self.net.to(device)

    def get_parameters(self):
        return self.net.parameters()

    def __call__(self, inputs: list[Molecule], num_results: int = 50) -> list[BackwardPredictionList]:
        raw_outputs = []
        for x in inputs:
            x = x.smiles
            arr = preprocess(x, self.fp_dim)
            arr = np.reshape(arr,[-1, arr.shape[0]])
            arr = torch.tensor(arr, dtype=torch.float32)
            arr = arr.to(self.device)
            preds = self.net(arr)
            preds = F.softmax(preds,dim=1)
            preds = preds.cpu()
            probs, idx = torch.topk(preds,k=num_results)
            # probs = F.softmax(probs,dim=1)
            rule_k = [self.idx2rules[id] for id in idx[0].numpy().tolist()]
            reactants = []
            scores = []
            templates = []
            for i , rule in enumerate(rule_k):
                out1 = []
                try:
                    out1 = rdchiralRunText(rule, x)
                    # out1 = rdchiralRunText(rule, Chem.MolToSmiles(Chem.MolFromSmarts(x)))
                    if len(out1) == 0: continue
                    # if len(out1) > 1: print("more than two reactants."),print(out1)
                    out1 = sorted(out1)
                    for reactant in out1:
                        reactants.append(reactant)
                        scores.append(probs[0][i].item()/len(out1))
                        templates.append(rule)
                # out1 = rdchiralRunText(x, rule)
                except ValueError:
                    pass
            if len(reactants) != 0: 
                reactants_d = defaultdict(list)
                for r, s, t in zip(reactants, scores, templates):
                    if '.' in r:
                        str_list = sorted(r.strip().split('.'))
                        reactants_d['.'.join(str_list)].append((s, t))
                    else:
                        reactants_d[r].append((s, t))

                reactants, scores, templates = merge(reactants_d)
                total = sum(scores)
                scores = [s / total for s in scores]
                raw_outputs.append((reactants, scores, templates))
            else:
                raw_outputs.append(([], [], []))
        return [
            process_raw_smiles_outputs(input=input, output_list=output[0], kwargs_list=[{'probability': score} for score in output[1]])
            for input, output in zip(inputs, raw_outputs)
        ]