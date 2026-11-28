import numpy as np
import pandas as pd
import polars as pl
import pytest
from numpy.testing import assert_array_equal, assert_equal
from rdkit.Chem import Mol, MolFromSmiles, MolToInchi

from skfp.preprocessing import MolFromInchiTransformer, MolToInchiTransformer


@pytest.fixture
def inchi_list(smiles_list):
    return [MolToInchi(MolFromSmiles(smi)) for smi in smiles_list]


def test_mol_from_inchi(inchi_list):
    mol_from_inchi = MolFromInchiTransformer()
    mols_list = mol_from_inchi.transform(inchi_list)

    assert_equal(len(mols_list), len(inchi_list))
    assert all(isinstance(x, Mol) for x in mols_list)


def test_mol_to_inchi(mols_list):
    mol_to_inchi = MolToInchiTransformer()
    inchi_list = mol_to_inchi.transform(mols_list)

    assert_equal(len(inchi_list), len(mols_list))
    assert all(isinstance(x, str) for x in inchi_list)


def test_mol_to_and_from_inchi(inchi_list):
    mol_from_inchi = MolFromInchiTransformer()
    mol_to_inchi = MolToInchiTransformer()

    mols_list = mol_from_inchi.transform(inchi_list)
    inchi_list_2 = mol_to_inchi.transform(mols_list)

    assert_equal(inchi_list_2, inchi_list)


def test_parallel_to_and_from_inchi(inchi_list):
    mol_from_inchi_seq = MolFromInchiTransformer()
    mol_from_inchi_parallel = MolFromInchiTransformer(n_jobs=-1)

    mol_to_inchi_seq = MolToInchiTransformer()
    mol_to_inchi_parallel = MolToInchiTransformer(n_jobs=-1)

    mols_list_seq = mol_from_inchi_seq.transform(inchi_list)
    mols_list_parallel = mol_from_inchi_parallel.transform(inchi_list)

    inchi_list_2_seq = mol_to_inchi_seq.transform(mols_list_seq)
    inchi_list_2_parallel = mol_to_inchi_parallel.transform(mols_list_parallel)

    assert_equal(inchi_list_2_seq, inchi_list)
    assert_equal(inchi_list_2_seq, inchi_list_2_parallel)


def test_from_invalid_inchi(inchi_list):
    invalid_inchi_list = ["noninchi", "invalid"]
    mol_from_inchi = MolFromInchiTransformer(valid_only=False)
    mols_list = mol_from_inchi.transform(inchi_list + invalid_inchi_list)

    mol_from_inchi = MolFromInchiTransformer(valid_only=True)
    mols_list_2 = mol_from_inchi.transform(inchi_list + invalid_inchi_list)

    assert_equal(len(mols_list), len(inchi_list) + len(invalid_inchi_list))
    assert_equal(len(mols_list_2), len(inchi_list))


def test_inchi_transform_x_y(inchi_list):
    labels = np.ones(len(inchi_list))

    mol_from_inchi = MolFromInchiTransformer(valid_only=True)
    mols, labels = mol_from_inchi.transform_x_y(inchi_list, labels)

    assert_equal(len(mols), len(inchi_list))
    assert_equal(len(labels), len(inchi_list))


@pytest.mark.parametrize(
    "labels",
    [
        [10, 20, 30],
        (10, 20, 30),
        np.array([10, 20, 30]),
        pd.Series([10, 20, 30], index=[100, 200, 300]),
        pl.Series([10, 20, 30]),
    ],
)
@pytest.mark.parametrize("valid_only", [True, False])
def test_from_invalid_inchi_with_different_label_types(labels, valid_only):
    inchi = [
        MolToInchi(MolFromSmiles("CCO")),
        "invalid",
        MolToInchi(MolFromSmiles("O")),
    ]
    mol_from_inchi = MolFromInchiTransformer(
        valid_only=valid_only, suppress_warnings=True
    )

    mols, selected_labels = mol_from_inchi.transform_x_y(inchi, labels)

    assert isinstance(selected_labels, np.ndarray)
    assert_equal(selected_labels.ndim, 1)

    if valid_only:
        assert_equal(len(mols), 2)
        assert_array_equal(selected_labels, [10, 30])
    else:
        assert_equal(len(mols), 3)
        assert_array_equal(selected_labels, [10, 20, 30])


@pytest.mark.parametrize(
    "labels",
    [
        np.array([[10], [20], [30]]),
        pd.DataFrame({"label": [10, 20, 30]}),
        pl.DataFrame({"label": [10, 20, 30]}),
    ],
)
def test_from_inchi_with_invalid_label_shape(labels):
    inchi = [
        MolToInchi(MolFromSmiles("CCO")),
        MolToInchi(MolFromSmiles("CC")),
        MolToInchi(MolFromSmiles("O")),
    ]
    mol_from_inchi = MolFromInchiTransformer()

    with pytest.raises(ValueError, match="only supports 1D target values"):
        mol_from_inchi.transform_x_y(inchi, labels)
