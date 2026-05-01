import numpy as np

try:
    from unimol_tools import UniMolRepr
    _UNIMOL_AVAILABLE = True
except ImportError:
    _UNIMOL_AVAILABLE = False


def smiles_to_unimol(smiles_list: list[str]) -> np.ndarray:
    """Extract Uni-Mol cls_repr for a list of SMILES. Returns (N, 512) float32."""
    if not _UNIMOL_AVAILABLE:
        raise ImportError("unimol_tools is not installed. Run: pip install unimol_tools")
    clf = UniMolRepr(data_type="molecule", remove_hs=False)
    reprs = clf.get_repr(smiles_list)
    if isinstance(reprs, dict):
        return np.array(reprs["cls_repr"], dtype=np.float32)
    elif isinstance(reprs, np.ndarray):
        return reprs.astype(np.float32)
    else:
        # list of dicts or list of arrays
        first = reprs[0]
        if isinstance(first, dict):
            return np.array([r["cls_repr"] for r in reprs], dtype=np.float32)
        return np.array(reprs, dtype=np.float32)
