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
from tqdm import tqdm
from rdchiral.main import rdchiralRunText, rdchiralRun
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
import pandas as pd
import json
import re
from rdchiral.main import rdchiralRunText, rdchiralRun
from rdchiral.initialization import rdchiralReaction, rdchiralReactants
from fast_filter.model import Net_orig
import pickle
import torch


def CDScore(p_mol, r_mols):
    p_atom_count = p_mol.GetNumAtoms()
    n_r_mols = len(r_mols)
    if n_r_mols == 1:
        return 0
    # r_atom_count = [r_mol.GetNumAtoms() for r_mol in r_mols]
    r_atom_count = [len([int(num[1:]) for num in re.findall(r':\d+', r_mol) if int(num[1:]) < 900]) for r_mol in r_mols]
    # print(r_atom_count)
    r_atom_count = [len([int(num[1:]) for num in re.findall(r':\d+', r_mol) if int(num[1:]) < 900]) for r_mol in r_mols]
    main_r = r_mols[np.argmax(r_atom_count)]
    if len(Chem.MolFromSmiles(main_r).GetAtoms()) >= p_atom_count:
        return 0
    MAE =  1 / n_r_mols * sum([abs(p_atom_count / n_r_mols - r_atom_count[i]) for i in range(n_r_mols)])
    # print(1 / (1 + MAE))
    return 1 / (1 + MAE) * p_atom_count

def CDScore_old(p_mol, r_mols):
    p_atom_count = p_mol.GetNumAtoms()
    n_r_mols = len(r_mols)
    r_atom_count = [r_mol.GetNumAtoms() for r_mol in r_mols]
    MAE =  1 / n_r_mols * sum([abs(p_atom_count / n_r_mols - r_atom_count[i]) for i in range(n_r_mols)])
    return 1 / (1 + MAE)

def ASScore(p_mol, r_mol_dict, in_stock):
    p_atom_count = p_mol.GetNumAtoms()
    r_mols = list(r_mol_dict.keys())
    r_atom_count = [len([int(num[1:]) for num in re.findall(r':\d+', r_mol) if int(num[1:]) < 900]) for r_mol in r_mols]
    main_r = r_mols[np.argmax(r_atom_count)]
    asscore = 0
    for k, v in r_mol_dict.items():
        if v in in_stock:
            add = len([int(num[1:]) for num in re.findall(r':\d+', k) if int(num[1:]) < 900])
            if len(Chem.MolFromSmiles(main_r).GetAtoms()) < p_atom_count:
                asscore += add
            else:
                asscore += add if add > 2 else 0
        if ('Mg' in v or 'Li' in v or 'Zn' in v) and v not in in_stock:
            asscore -= 10
    return asscore

def RDScore(p_mol, r_mols):
    p_ring_count = p_mol.GetRingInfo().NumRings()
    r_rings_s = [r_mol.GetRingInfo().AtomRings() for r_mol in r_mols]
    r_ring_count = 0
    for r_rings, r_mol in zip(r_rings_s, r_mols):
        for r_ring in r_rings:
            mapnums = [r_mol.GetAtomWithIdx(i).GetAtomMapNum() for i in r_ring]
            symbols = [r_mol.GetAtomWithIdx(i).GetSymbol() for i in r_ring]
            if 'B' in symbols or 'Si' in symbols:
                continue
            if min(mapnums) < 900:
                r_ring_count += 1
    if p_ring_count > r_ring_count:
        return 1
    else:
        return 0
    
def canonical_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(Chem.MolFromSmiles(Chem.MolToSmiles(mol)))

def smiles_to_fingerprint(smiles, fp_length=2048, radius=2):
    mol = Chem.MolFromSmiles(smiles)
    return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=fp_length)).reshape(1, -1)


def canonical_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(Chem.MolFromSmiles(Chem.MolToSmiles(mol)))

class SimpRetro(BackwardReactionModel):
    def __init__(
        self,
        model_dir: Union[str, Path],
        device):
        self.templates_raw = json.load(open(model_dir)) # list(set(pd.read_csv(model_dir, sep='\t').retro_template.values))
        print(f'Total Number of Templates: {len(self.templates_raw)}')
        self.template_list = []
        for i, l in tqdm(enumerate(self.templates_raw), desc='loading templates'):
            rule= l.strip()
            self.template_list.append(rdchiralReaction(rule))
        self.fingerprint_base = pickle.load(open('fast_filter/fingerprint_base.pkl', 'rb')) if os.path.exists('fast_filter/fingerprint_base.pkl') else {}
        self.instock_list = set(open('emolecules.txt').read().split('\n'))
        self.template_fps = np.array([self.fingerprint_base.get(template) for template in self.templates_raw])
        self.filter = Net_orig()
        self.filter.load_state_dict(torch.load('fast_filter/model_smoothbce.pth', map_location='cpu'))

    def get_parameters(self):
        return 0

    def __call__(self, inputs: list[Molecule], num_results: int = 50) -> list[BackwardPredictionList]:
        raw_outputs = []
        w1, w2, w3, w4 = 0.1, 0.2, 0.5, 0
        for x in inputs:
            results = {}
            result_set = set([])
            p_mol = Chem.MolFromSmiles(x.smiles)
            p_mol_rdchiral = rdchiralReactants(x.smiles)
            valid_template_id = []
            for idx, (template, template_raw) in enumerate(zip(self.template_list, self.templates_raw)):
                mapped_curr_results = rdchiralRun(template, p_mol_rdchiral, keep_mapnums=True)
                for r in mapped_curr_results:
                    canonical_r = canonical_smiles(r)
                    canonical_r_dict = {r_: canonical_smiles(r_) for r_ in r.split('.')}
                    if idx not in valid_template_id:
                        valid_template_id.append(idx)
                    if canonical_r in result_set:
                        continue
                    result_set.add(canonical_r)
                    r_mols = [Chem.MolFromSmiles(r_) for r_ in r.split('.')]
                    threshold = 0.2
                    rdscore = RDScore(p_mol, r_mols)
                    score = 1 * (w1 * CDScore(p_mol, r.split('.')) + w2 * ASScore(p_mol, canonical_r_dict, self.instock_list) + w3 * rdscore + w4 * 1 / len(mapped_curr_results))
                    results[canonical_r] = (score, template_raw, idx, rdscore)
            valid_temp_fps = self.template_fps[valid_template_id]
            p_fp = smiles_to_fingerprint(x.smiles)
            try:
                data = torch.tensor(np.concatenate([valid_temp_fps.squeeze(), np.repeat(p_fp, len(valid_temp_fps), axis=0)], axis=1), dtype=torch.float32) # .to('cuda')
                with torch.no_grad():
                    pred = self.filter(data).squeeze().cpu().numpy()
                validated_results = {}
                for i, (k, v) in enumerate(results.items()):
                    if pred[valid_template_id.index(v[2])] > threshold or v[-1]:
                        validated_results[k] = (v[0], v[1], v[2], pred[valid_template_id.index(v[2])])
                    else:
                        pass 
            except:
                print(valid_temp_fps)
                validated_results = {}
            results = sorted(validated_results.items(), key=lambda item: item[1][0] + 0.001 * item[1][-1], reverse=True)[:50]
            if not len(results) == 0:
                reactants, scores = zip(*results)
                templates = [t[1] for t in scores]
                scores = [s[0] for s in scores]
                scores = [np.exp(s) for s in scores]
                scores = [s / sum(scores) for s in scores]
                raw_outputs.append((reactants, scores))
            else:
                raw_outputs.append(([], []))
        return [
            process_raw_smiles_outputs(input=input, output_list=output[0], kwargs_list=[{'probability': score} for score in output[1]])
            for input, output in zip(inputs, raw_outputs)
        ]
