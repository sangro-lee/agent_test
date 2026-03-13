from typing import List

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

try:
    from rdkit.Chem import rdFingerprintGenerator
except ImportError:  # pragma: no cover
    rdFingerprintGenerator = None


def _bitvect_to_array(bitvect, n_bits: int) -> np.ndarray:
    arr = np.zeros((n_bits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(bitvect, arr)
    return arr


def smiles_to_fp(
    smiles_list: List[str],
    bits: int = 2048,
    radius: int = 2,
    use_chirality: bool = False,
) -> np.ndarray:
    fps = np.zeros((len(smiles_list), bits), dtype=np.float32)

    generator = None
    if rdFingerprintGenerator is not None and hasattr(rdFingerprintGenerator, "GetMorganGenerator"):
        generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=radius,
            fpSize=bits,
            includeChirality=use_chirality,
        )

    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            continue

        try:
            if generator is not None:
                fp = generator.GetFingerprint(mol)
            else:
                fp = AllChem.GetMorganFingerprintAsBitVect(
                    mol,
                    radius=radius,
                    nBits=bits,
                    useChirality=use_chirality,
                )
            fps[i] = _bitvect_to_array(fp, bits)
        except Exception:
            # Keep zero-vector for failed featurization.
            continue

    return fps
