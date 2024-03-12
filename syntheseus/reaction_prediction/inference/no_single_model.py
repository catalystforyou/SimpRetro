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
from rdkit import RDLogger
import pandas as pd
import json
import re
from rdchiral.main import rdchiralRunText, rdchiralRun
from rdchiral.initialization import rdchiralReaction, rdchiralReactants



def CDScore(p_mol, r_mols):
    p_atom_count = p_mol.GetNumAtoms()
    n_r_mols = len(r_mols)
    if n_r_mols == 1:
        return 0
    # r_atom_count = [r_mol.GetNumAtoms() for r_mol in r_mols]
    r_atom_count = [len([int(num[1:]) for num in re.findall(r':\d+', r_mol) if int(num[1:]) < 900]) for r_mol in r_mols]
    # print(r_atom_count)
    MAE =  1 / n_r_mols * sum([abs(p_atom_count / n_r_mols - r_atom_count[i]) for i in range(n_r_mols)])
    # print(1 / (1 + MAE))
    return 1 / (1 + MAE)

def CDScore_old(p_mol, r_mols):
    p_atom_count = p_mol.GetNumAtoms()
    n_r_mols = len(r_mols)
    r_atom_count = [r_mol.GetNumAtoms() for r_mol in r_mols]
    MAE =  1 / n_r_mols * sum([abs(p_atom_count / n_r_mols - r_atom_count[i]) for i in range(n_r_mols)])
    return 1 / (1 + MAE)

def ASScore(r_mols, in_stock):
    return sum([1 if r_mol in in_stock else 0 for r_mol in r_mols]) / len(r_mols)

def RDScore(p_mol, r_mols):
    p_ring_count = p_mol.GetRingInfo().NumRings()
    r_ring_count = sum([r_mol.GetRingInfo().NumRings() for r_mol in r_mols])
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

class NoModel(BackwardReactionModel):
    def __init__(
        self,
        model_dir: Union[str, Path],
        device):
        """Initializes the NeuralSym model wrapper."""
        self.templates_raw = json.load(open(model_dir)) # list(set(pd.read_csv(model_dir, sep='\t').retro_template.values))
        print(f'Total Number of Templates: {len(self.templates_raw)}')
        self.template_list = []
        for i, l in tqdm(enumerate(self.templates_raw), desc='loading templates'):
            rule= l.strip()
            self.template_list.append(rdchiralReaction(rule))
        self.instock_list = set(open('emolecules.txt').read().split('\n'))

    def get_parameters(self):
        return 0

    def __call__(self, inputs: list[Molecule], num_results: int = 50) -> list[BackwardPredictionList]:
        raw_outputs = []
        w1, w2, w3, w4 = 1, 2, 0.5, 1
        for x in inputs:
            results = {}
            result_set = set([])
            p_mol = Chem.MolFromSmiles(x.smiles)
            p_mol_rdchiral = rdchiralReactants(x.smiles)
            for template, template_raw in zip(self.template_list, self.templates_raw):
                mapped_curr_results = rdchiralRun(template, p_mol_rdchiral, keep_mapnums=True)
                for r in mapped_curr_results:
                    canonical_r = canonical_smiles(r)
                    if canonical_r in result_set:
                        continue
                    result_set.add(canonical_r)
                    r_mols = [Chem.MolFromSmiles(r_) for r_ in canonical_r.split('.')]
                    r_smls = canonical_r.split('.')
                    score = 1 * (w1 * CDScore(p_mol, r.split('.')) + w2 * ASScore(r_smls, self.instock_list) + w3 * RDScore(p_mol, r_mols) + w4 * 1 / len(r))
                    # score = 1 * (w1 * CDScore_old(p_mol, r_mols) + w2 * ASScore(r_smls, self.instock_list) + w3 * RDScore(p_mol, r_mols) + w4 * 1 / len(r))
                    results[canonical_r] = (score, template_raw)
            results = sorted(results.items(), key=lambda item: item[1][0], reverse=True)
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