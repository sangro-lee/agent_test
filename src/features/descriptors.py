from typing import List

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors


_DESC_FUNCS = [func for _, func in Descriptors._descList]


def smiles_to_descriptors(smiles_list: List[str]) -> np.ndarray:
    rows = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            rows.append([0.0] * len(_DESC_FUNCS))
            continue

        vals = []
        for fn in _DESC_FUNCS:
            try:
                vals.append(float(fn(mol)))
            except Exception:
                vals.append(0.0)
        rows.append(vals)

    arr = np.array(rows, dtype=np.float32)
    arr[~np.isfinite(arr)] = 0.0
    return arr
