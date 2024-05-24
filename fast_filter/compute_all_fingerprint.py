from rdkit import Chem
import numpy as np
import pickle
from rdkit.Chem import AllChem
import json
from tqdm import tqdm


rules = json.load(open('USPTO_full_train_templates_1.json'))

database = pickle.load(open('fingerprint_base.pkl', 'rb'))

def smarts_to_fingerprint(smarts):
    rxn = AllChem.ReactionFromSmarts(smarts)
    return np.concatenate([np.array(AllChem.CreateDifferenceFingerprintForReaction(rxn).ToList()).reshape(1, -1), np.array(AllChem.CreateStructuralFingerprintForReaction(rxn).ToList()).reshape(1, -1)], axis=1)

for rule in tqdm(rules):
    if rule not in database:
        database[rule] = smarts_to_fingerprint(rule)

pickle.dump(database, open('fingerprint_base.pkl', 'wb'))