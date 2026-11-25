# ruff: noqa: INP001
import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from rdkit import Chem

from skfp.fingerprints.rdkit_2d_desc import _get_rdkit_2d_descriptors
from skfp.utils import no_rdkit_logs, run_in_parallel

N_BINS = 512
SAMPLE_SIZE = 1_000_000
BATCH_SIZE = 1000
RANDOM_SEED = 42
# integer bins are shifted by this offset, since some count descriptors can be
# negative, e.g. NumValenceElectrons for highly charged bare ions like [U+6]
COUNT_OFFSET = 16

OUTPUT_PATH = (
    Path(__file__).resolve().parents[1]
    / "skfp"
    / "fingerprints"
    / "data"
    / "rdkit_2d_cdf_molpile.json"
)


def _get_count_descriptors_mask() -> np.ndarray:
    """Check which descriptors are integer counts, based on their return type."""
    mol = Chem.MolFromSmiles("CCO")
    return np.array(
        [isinstance(func(mol), int) for _, func in _get_rdkit_2d_descriptors()]
    )


def _calculate_descriptors(smiles_batch: list[str]) -> np.ndarray:
    """Calculate descriptors, with NaN rows for molecules that could not be parsed."""
    descriptors = _get_rdkit_2d_descriptors()
    values = np.full((len(smiles_batch), len(descriptors)), np.nan)

    with no_rdkit_logs():
        for mol_idx, smiles in enumerate(smiles_batch):
            mol = Chem.MolFromSmiles(smiles) if smiles else None
            if mol is None:
                continue

            for desc_idx, (_, func) in enumerate(descriptors):
                try:
                    values[mol_idx, desc_idx] = func(mol)
                except Exception:  # noqa: BLE001, S112
                    continue

    return values


def _calculate_continuous_descriptors(
    smiles_batch: list[str], is_count: np.ndarray
) -> np.ndarray:
    return _calculate_descriptors(smiles_batch)[:, ~is_count]


def _count_batch(
    smiles_batch: list[str], thresholds: dict[str, np.ndarray]
) -> tuple[list[np.ndarray], np.ndarray, int]:
    """
    Calculate histogram counts for each descriptor. Count descriptors get a bin per
    integer value. Continuous ones use right-inclusive bins ``(t[i-1], t[i]]``, with
    an additional last bin for values above the largest threshold.
    """
    values = _calculate_descriptors(smiles_batch)
    counts = []
    maxima = np.full(values.shape[1], -np.inf)

    for desc_idx, (name, _) in enumerate(_get_rdkit_2d_descriptors()):
        finite = values[np.isfinite(values[:, desc_idx]), desc_idx]

        if name in thresholds:
            bins = np.searchsorted(thresholds[name], finite, side="left")
            counts.append(np.bincount(bins, minlength=len(thresholds[name]) + 1))
        else:
            if finite.size and finite.min() < -COUNT_OFFSET:
                raise ValueError(f"Value below -{COUNT_OFFSET} for descriptor: {name}")
            counts.append(np.bincount(finite.astype(np.int64) + COUNT_OFFSET))

        if finite.size:
            maxima[desc_idx] = finite.max()

    num_invalid = int(np.isnan(values).all(axis=1).sum())
    return counts, maxima, num_invalid


def _add_counts(total: np.ndarray, counts: np.ndarray) -> np.ndarray:
    if len(counts) > len(total):
        total, counts = counts.copy(), total
    total[: len(counts)] += counts
    return total


def _shard_path(work_dir: Path, shard: int, num_shards: int) -> Path:
    return work_dir / f"counts_{shard:05d}_of_{num_shards:05d}.npz"


def compute_bins(
    parquet_file: pq.ParquetFile, work_dir: Path, sample_size: int, n_jobs: int
) -> None:
    """Estimate bin thresholds of continuous descriptors on a random sample."""
    path = work_dir / "bins.npz"
    if path.exists():
        print(f"Bins already computed: {path}")
        return

    num_rows = parquet_file.metadata.num_rows
    rng = np.random.default_rng(RANDOM_SEED)
    indices = np.sort(
        rng.choice(num_rows, size=min(sample_size, num_rows), replace=False)
    )

    smiles: list[str] = []
    offset = 0
    for row_group in range(parquet_file.num_row_groups):
        next_offset = offset + parquet_file.metadata.row_group(row_group).num_rows
        start, end = np.searchsorted(indices, [offset, next_offset])
        if start < end:
            column = parquet_file.read_row_group(row_group, columns=["SMILES"])[0]
            smiles.extend(column.take(indices[start:end] - offset).to_pylist())
        offset = next_offset

    print(f"Calculating bins on {len(smiles)} sampled molecules", flush=True)
    is_count = _get_count_descriptors_mask()
    values = np.concatenate(
        run_in_parallel(
            _calculate_continuous_descriptors,
            smiles,
            n_jobs=n_jobs,
            batch_size=BATCH_SIZE,
            is_count=is_count,
        )
    )

    names = [name for name, _ in _get_rdkit_2d_descriptors()]
    continuous_names = [
        name for name, count in zip(names, is_count, strict=True) if not count
    ]
    quantiles = np.linspace(0, 1, N_BINS)

    thresholds = {}
    for desc_idx, name in enumerate(continuous_names):
        finite = values[np.isfinite(values[:, desc_idx]), desc_idx]
        if not finite.size:
            raise ValueError(f"No finite sample values for descriptor: {name}")
        thresholds[name] = np.unique(np.quantile(finite, quantiles, method="higher"))

    work_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = work_dir / "bins.tmp.npz"
    np.savez(tmp_path, **thresholds)
    tmp_path.replace(path)
    print(f"Saved bins of {len(thresholds)} continuous descriptors: {path}")


def count_shard(
    parquet_file: pq.ParquetFile,
    work_dir: Path,
    shard: int,
    num_shards: int,
    n_jobs: int,
) -> None:
    """Calculate histogram counts on every ``num_shards``-th row group."""
    path = _shard_path(work_dir, shard, num_shards)
    if path.exists():
        print(f"Shard {shard} already computed: {path}")
        return

    with np.load(work_dir / "bins.npz") as bins:
        thresholds = {name: bins[name] for name in bins.files}

    names = [name for name, _ in _get_rdkit_2d_descriptors()]
    total_counts = [np.zeros(0, dtype=np.int64) for _ in names]
    total_maxima = np.full(len(names), -np.inf)
    num_molecules = 0
    num_invalid = 0

    row_groups = range(shard, parquet_file.num_row_groups, num_shards)
    for i, row_group in enumerate(row_groups, start=1):
        table = parquet_file.read_row_group(row_group, columns=["SMILES"])
        smiles = table[0].to_pylist()

        results = run_in_parallel(
            _count_batch,
            smiles,
            n_jobs=n_jobs,
            batch_size=BATCH_SIZE,
            thresholds=thresholds,
        )
        for counts, maxima, invalid in results:
            total_counts = [
                _add_counts(total, new)
                for total, new in zip(total_counts, counts, strict=True)
            ]
            total_maxima = np.maximum(total_maxima, maxima)
            num_invalid += invalid

        num_molecules += len(smiles)
        print(
            f"Shard {shard}/{num_shards}: row group {i}/{len(row_groups)}, "
            f"{num_molecules} molecules",
            flush=True,
        )

    tmp_path = path.with_suffix(".tmp.npz")
    np.savez(
        tmp_path,
        maxima=total_maxima,
        num_molecules=num_molecules,
        num_invalid=num_invalid,
        **{
            f"counts_{name}": counts
            for name, counts in zip(names, total_counts, strict=True)
        },
    )
    tmp_path.replace(path)
    print(f"Saved shard {shard}: {path}")


def _get_cdf(
    counts: np.ndarray, thresholds: np.ndarray | None, maximum: float
) -> tuple[np.ndarray, np.ndarray]:
    """Turn histogram counts into CDF thresholds and their probabilities."""
    probabilities = np.cumsum(counts) / counts.sum()

    if thresholds is None:
        # count descriptor, bins are consecutive integers
        thresholds = np.arange(len(counts), dtype=np.float64) - COUNT_OFFSET
    else:
        # continuous descriptor, last bin has values above the largest threshold
        thresholds = np.append(thresholds, maximum)

    nonempty = counts > 0
    thresholds = thresholds[nonempty]
    probabilities = probabilities[nonempty]

    if len(thresholds) > N_BINS:
        quantiles = np.linspace(0, 1, N_BINS)
        selected = np.unique(np.searchsorted(probabilities, quantiles, side="left"))
        selected = selected[selected < len(thresholds)]
        thresholds = thresholds[selected]
        probabilities = probabilities[selected]

    return thresholds, probabilities


def merge_shards(
    parquet_file: pq.ParquetFile, work_dir: Path, num_shards: int, output: Path
) -> None:
    """Sum histogram counts from all shards and save the resulting CDFs."""
    with np.load(work_dir / "bins.npz") as bins:
        thresholds = {name: bins[name] for name in bins.files}

    names = [name for name, _ in _get_rdkit_2d_descriptors()]
    total_counts = [np.zeros(0, dtype=np.int64) for _ in names]
    total_maxima = np.full(len(names), -np.inf)
    num_molecules = 0
    num_invalid = 0

    for shard in range(num_shards):
        with np.load(_shard_path(work_dir, shard, num_shards)) as data:
            total_counts = [
                _add_counts(total, data[f"counts_{name}"])
                for total, name in zip(total_counts, names, strict=True)
            ]
            total_maxima = np.maximum(total_maxima, data["maxima"])
            num_molecules += int(data["num_molecules"])
            num_invalid += int(data["num_invalid"])

    if num_molecules != parquet_file.metadata.num_rows:
        raise ValueError(
            f"Processed {num_molecules} molecules, "
            f"expected {parquet_file.metadata.num_rows}"
        )

    lines = []
    for name, counts, maximum in zip(names, total_counts, total_maxima, strict=True):
        if counts.sum() == 0:
            raise ValueError(f"No finite values for descriptor: {name}")

        cdf = _get_cdf(counts, thresholds.get(name), maximum)
        cdf_json = json.dumps([x.tolist() for x in cdf], allow_nan=False)
        lines.append(f"{json.dumps(name)}: {cdf_json}")

    # one descriptor per line, to keep the file diffable
    output.write_text("{\n" + ",\n".join(lines) + "\n}\n", encoding="utf-8")
    print(
        f"Saved CDF data for {len(lines)} descriptors, based on {num_molecules} "
        f"molecules ({num_invalid} invalid): {output}"
    )


def main() -> None:  # noqa: D103
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "stage", nargs="?", default="all", choices=["all", "bins", "count", "merge"]
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Parquet file with SMILES column. By default, the full MolPILE dataset "
        "is downloaded from HuggingFace Hub.",
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--work-dir", type=Path, default=Path("molpile_cdf"))
    parser.add_argument("--sample-size", type=int, default=SAMPLE_SIZE)
    parser.add_argument("--num-shards", type=int, default=100)
    parser.add_argument(
        "--shard", type=int, help="Shard to count. By default, all are counted."
    )
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()

    source = args.input or hf_hub_download(
        repo_id="scikit-fingerprints/MolPILE",
        repo_type="dataset",
        filename="molpile_smiles.parquet",
    )
    parquet_file = pq.ParquetFile(source)

    if args.stage in {"all", "bins"}:
        compute_bins(parquet_file, args.work_dir, args.sample_size, args.n_jobs)

    if args.stage in {"all", "count"}:
        shards = range(args.num_shards) if args.shard is None else [args.shard]
        for shard in shards:
            count_shard(
                parquet_file, args.work_dir, shard, args.num_shards, args.n_jobs
            )

    if args.stage in {"all", "merge"}:
        merge_shards(parquet_file, args.work_dir, args.num_shards, args.output)


if __name__ == "__main__":
    main()
