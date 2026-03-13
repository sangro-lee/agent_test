"""
Molecular graph featurization.

Includes two featurization schemes:
  - Default (legacy): small feature set for AttentiveFP / GCN.
  - SME (Substructure Mask Explanation, Wu et al. 2023): 40-dim atom features
    + integer edge-type (for RGCN) + per-atom substructure mask (smask).

SME reference implementation: Substructure-Mask-Explanation/MaskGNN_interpretation/build_data.py
Ported from DGL → PyTorch Geometric; DGL dependency NOT required.
"""

from typing import List, Optional, Tuple

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import BRICS
from rdkit.Chem.Scaffolds import MurckoScaffold
from torch_geometric.data import Data, Dataset

# ---------------------------------------------------------------------------
# Default (legacy) featurization  — kept for AttentiveFP compatibility
# ---------------------------------------------------------------------------

_HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]
_BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]

# Default atom-feature dimension: 5 scalar + 5 hybridization = 10
DEFAULT_NODE_DIM = 10
# Default edge-feature dimension: 4 bond-type + 2 scalar = 6
DEFAULT_EDGE_DIM = 6


def _atom_features(atom: Chem.rdchem.Atom) -> List[float]:
    hybrid_onehot = [1.0 if atom.GetHybridization() == h else 0.0 for h in _HYBRIDIZATIONS]
    return [
        float(atom.GetAtomicNum()),
        float(atom.GetDegree()),
        float(atom.GetFormalCharge()),
        float(atom.IsInRing()),
        float(atom.GetIsAromatic()),
        *hybrid_onehot,
    ]


def _bond_features(bond: Chem.rdchem.Bond) -> List[float]:
    bond_onehot = [1.0 if bond.GetBondType() == b else 0.0 for b in _BOND_TYPES]
    return [
        *bond_onehot,
        float(bond.GetIsConjugated()),
        float(bond.IsInRing()),
    ]


def mol_to_graph(mol: Chem.Mol) -> Data:
    """Default mol→graph (AttentiveFP compatible)."""
    if mol is None or mol.GetNumAtoms() == 0:
        x = torch.zeros((1, DEFAULT_NODE_DIM), dtype=torch.float32)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, DEFAULT_EDGE_DIM), dtype=torch.float32)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    node_feats = [_atom_features(atom) for atom in mol.GetAtoms()]
    x = torch.tensor(node_feats, dtype=torch.float32)

    edges, edge_feats = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf = _bond_features(bond)
        edges += [[i, j], [j, i]]
        edge_feats += [bf, bf]

    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_feats, dtype=torch.float32)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, DEFAULT_EDGE_DIM), dtype=torch.float32)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def smiles_to_graph(smiles: str) -> Data:
    mol = Chem.MolFromSmiles(str(smiles))
    return mol_to_graph(mol)


# ---------------------------------------------------------------------------
# SME featurization  (Wu et al. 2023)
# ---------------------------------------------------------------------------

_SME_SYMBOLS = ['B', 'C', 'N', 'O', 'F', 'Si', 'P', 'S',
                 'Cl', 'As', 'Se', 'Br', 'Te', 'I', 'At', 'other']
_SME_DEGREES = [0, 1, 2, 3, 4, 5, 6]
_SME_HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
    'other',
]
_SME_H_COUNT = [0, 1, 2, 3, 4]
_SME_CIP = ['R', 'S']
_SME_STEREO = ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"]

# Atom feature dim: 16 + 7 + 2 + 6 + 1 + 5 + 2 + 1 = 40
SME_NODE_DIM = 40


def _one_of_k_unk(x, allowable_set: list) -> List[int]:
    """One-hot, unknown → last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [int(x == s) for s in allowable_set]


def _one_of_k(x, allowable_set: list) -> List[int]:
    """One-hot, raise if unknown."""
    if x not in allowable_set:
        raise ValueError(f"{x} not in {allowable_set}")
    return [int(x == s) for s in allowable_set]


def sme_atom_features(atom: Chem.rdchem.Atom, use_chirality: bool = True) -> np.ndarray:
    """40-dim atom features matching SME (Wu et al. 2023) build_data.atom_features()."""
    feats: List = (
        _one_of_k_unk(atom.GetSymbol(), _SME_SYMBOLS)           # 16
        + _one_of_k(atom.GetDegree(), _SME_DEGREES)             # 7
        + [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()]  # 2
        + _one_of_k_unk(atom.GetHybridization(), _SME_HYBRIDIZATIONS)  # 6
        + [int(atom.GetIsAromatic())]                            # 1
        + _one_of_k_unk(atom.GetTotalNumHs(), _SME_H_COUNT)     # 5
    )
    if use_chirality:
        try:
            feats += _one_of_k_unk(atom.GetProp('_CIPCode'), _SME_CIP)  # 2
            feats += [int(atom.HasProp('_ChiralityPossible'))]            # 1
        except KeyError:
            feats += [0, 0, int(atom.HasProp('_ChiralityPossible'))]
    return np.array(feats, dtype=np.float32)


def sme_etype(bond: Chem.rdchem.Bond, use_chirality: bool = True) -> int:
    """Integer edge-type for RGCN (0-63).

    Encoding matches SME build_data.etype_features():
      index = bond_type_idx * 1 + is_conjugated * 4 + is_in_ring * 8 + stereo_idx * 16
    """
    bt = bond.GetBondType()
    bt_order = [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ]
    a = next((i for i, b in enumerate(bt_order) if bt == b), 0)
    b = int(bond.GetIsConjugated())
    c = int(bond.IsInRing())
    index = a + b * 4 + c * 8
    if use_chirality:
        stereo_str = str(bond.GetStereo())
        d = _SME_STEREO.index(stereo_str) if stereo_str in _SME_STEREO else 0
        index += d * 16
    return index  # max 63


# Maximum num_rels for RGCN (3 + 4 + 8 + 3*16 = 63, use 64 for safety)
SME_NUM_RELS = 64

# ---------------------------------------------------------------------------
# Substructure mask helpers
# ---------------------------------------------------------------------------


def brics_mask(mol: Chem.Mol) -> np.ndarray:
    """Binary per-atom mask: 1 if the atom belongs to any BRICS fragment."""
    n = mol.GetNumAtoms()
    mask = np.zeros(n, dtype=np.float32)
    try:
        frags = BRICS.BRICSDecompose(mol, returnMols=False)
        # Get atom indices in each fragment via substructure match
        for frag_smi in frags:
            frag_mol = Chem.MolFromSmiles(frag_smi)
            if frag_mol is None:
                continue
            matches = mol.GetSubstructMatches(frag_mol)
            for match in matches:
                for idx in match:
                    mask[idx] = 1.0
    except Exception:
        pass
    # Fallback: if no fragment found, all atoms get mask=1
    if mask.sum() == 0:
        mask[:] = 1.0
    return mask


def murcko_mask(mol: Chem.Mol) -> np.ndarray:
    """Binary per-atom mask: 1 if the atom belongs to Bemis-Murcko scaffold."""
    n = mol.GetNumAtoms()
    mask = np.zeros(n, dtype=np.float32)
    try:
        scaffold_smi = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        scaffold_mol = Chem.MolFromSmiles(scaffold_smi)
        if scaffold_mol is not None:
            matches = mol.GetSubstructMatches(scaffold_mol)
            for match in matches:
                for idx in match:
                    mask[idx] = 1.0
    except Exception:
        pass
    if mask.sum() == 0:
        mask[:] = 1.0
    return mask


def _get_smask(mol: Chem.Mol, smask_type: str) -> np.ndarray:
    if smask_type == "brics":
        return brics_mask(mol)
    elif smask_type == "murcko":
        return murcko_mask(mol)
    else:  # "none" or any other → uniform mask
        return np.ones(mol.GetNumAtoms(), dtype=np.float32)


# ---------------------------------------------------------------------------
# SME mol → PyG graph
# ---------------------------------------------------------------------------

def sme_mol_to_graph(
    mol: Chem.Mol,
    smask_type: str = "brics",
    use_chirality: bool = True,
) -> Data:
    """Convert RDKit mol to PyG Data using SME featurization.

    Returns Data with:
      x          : (N, 40) float  — SME atom features
      edge_index : (2, 2E) long
      edge_attr  : (2E, 6) float  — default bond features (for AttentiveFP compat.)
      edge_type  : (2E,) long     — integer bond type for RGCNConv (0-63)
      smask      : (N, 1) float   — substructure mask (1 = in substructure)
    """
    if mol is None or mol.GetNumAtoms() == 0:
        x = torch.zeros((1, SME_NODE_DIM), dtype=torch.float32)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, DEFAULT_EDGE_DIM), dtype=torch.float32)
        edge_type = torch.zeros((0,), dtype=torch.long)
        smask = torch.ones((1, 1), dtype=torch.float32)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                    edge_type=edge_type, smask=smask)

    # Node features
    node_feats = [sme_atom_features(atom, use_chirality) for atom in mol.GetAtoms()]
    x = torch.tensor(np.stack(node_feats), dtype=torch.float32)

    # Substructure mask
    mask_arr = _get_smask(mol, smask_type)
    smask = torch.tensor(mask_arr, dtype=torch.float32).unsqueeze(1)  # (N, 1)

    # Edges
    edges, edge_feats, etypes = [], [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf = _bond_features(bond)
        et = sme_etype(bond, use_chirality)
        edges += [[i, j], [j, i]]
        edge_feats += [bf, bf]
        etypes += [et, et]

    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_feats, dtype=torch.float32)
        edge_type = torch.tensor(etypes, dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, DEFAULT_EDGE_DIM), dtype=torch.float32)
        edge_type = torch.zeros((0,), dtype=torch.long)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                edge_type=edge_type, smask=smask)


def sme_smiles_to_graph(smiles: str, smask_type: str = "brics") -> Data:
    mol = Chem.MolFromSmiles(str(smiles))
    return sme_mol_to_graph(mol, smask_type=smask_type)


# ---------------------------------------------------------------------------
# Dataset classes
# ---------------------------------------------------------------------------

class MolDataset(Dataset):
    """Dataset using default (legacy) featurization."""

    def __init__(self, smiles_list: List[str], labels: Optional[List[float]] = None):
        super().__init__(root=None)
        self.smiles_list = [str(s) for s in smiles_list]
        self.labels = labels

    def len(self) -> int:
        return len(self.smiles_list)

    def get(self, idx: int) -> Data:
        data = smiles_to_graph(self.smiles_list[idx])
        if self.labels is not None:
            data.y = torch.tensor([float(self.labels[idx])], dtype=torch.float32)
        data.idx = torch.tensor([idx], dtype=torch.long)
        return data

    def get_pair(self, idx: int) -> Tuple[str, Optional[float]]:
        label = None if self.labels is None else float(self.labels[idx])
        return self.smiles_list[idx], label


class SMEMolDataset(Dataset):
    """Dataset using SME featurization (40-dim atoms + smask + edge_type)."""

    def __init__(
        self,
        smiles_list: List[str],
        labels: Optional[List[float]] = None,
        smask_type: str = "brics",
        use_chirality: bool = True,
    ):
        super().__init__(root=None)
        self.smiles_list = [str(s) for s in smiles_list]
        self.labels = labels
        self.smask_type = smask_type
        self.use_chirality = use_chirality

    def len(self) -> int:
        return len(self.smiles_list)

    def get(self, idx: int) -> Data:
        data = sme_smiles_to_graph(self.smiles_list[idx], self.smask_type)
        if self.labels is not None:
            data.y = torch.tensor([float(self.labels[idx])], dtype=torch.float32)
        data.idx = torch.tensor([idx], dtype=torch.long)
        return data

    def get_pair(self, idx: int) -> Tuple[str, Optional[float]]:
        label = None if self.labels is None else float(self.labels[idx])
        return self.smiles_list[idx], label
