#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Recreation of the UCI "Human Activity Recognition Using Smartphones" study
==========================================================================

Reference paper
---------------
Davide Anguita, Alessandro Ghio, Luca Oneto, Xavier Parra and Jorge L. Reyes-Ortiz.
"A Public Domain Dataset for Human Activity Recognition Using Smartphones".
21st European Symposium on Artificial Neural Networks, Computational Intelligence
and Machine Learning (ESANN 2013), Bruges (Belgium), 24-26 April 2013, pp. 437-442.

Dataset
-------
UCI Machine Learning Repository, id 240:
https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
"""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

# --------------------------------------------------------------------------- #
# 0. Third-party dependencies - installed on the fly so that this single file
#    can run on a bare Python installation.
# --------------------------------------------------------------------------- #
_REQUIRED_PACKAGES = (("numpy", "numpy"), ("scipy", "scipy"), ("sklearn", "scikit-learn"))


def ensure_dependencies(auto_install: bool = True) -> None:
    """Make sure numpy / scipy / scikit-learn are importable."""
    missing = [pkg for module, pkg in _REQUIRED_PACKAGES if importlib.util.find_spec(module) is None]
    if not missing:
        return
    if not auto_install:
        sys.exit(f"Missing packages: {', '.join(missing)} - install them with 'pip install {' '.join(missing)}'.")
    print(f"[setup] installing missing packages: {' '.join(missing)} ...", flush=True)
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "--quiet", *missing]
    )


ensure_dependencies()

import numpy as np  # noqa: E402  (imported after the dependency bootstrap)
from sklearn.base import clone  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_score  # noqa: E402
from sklearn.multiclass import OneVsRestClassifier  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

# --------------------------------------------------------------------------- #
# 1. Constants taken from the paper / dataset README
# --------------------------------------------------------------------------- #
PAPER = {
    "authors": "D. Anguita, A. Ghio, L. Oneto, X. Parra, J. L. Reyes-Ortiz",
    "title": "A Public Domain Dataset for Human Activity Recognition Using Smartphones",
    "venue": "ESANN 2013, Bruges (Belgium), 24-26 April 2013, pp. 437-442",
    "dataset": "UCI ML Repository id 240 (https://archive.ics.uci.edu/dataset/240/...)",
    "n_subjects": 30,
    "n_features": 561,
    "sampling_rate_hz": 50,
    "window_seconds": 2.56,
    "window_samples": 128,
    "overlap": 0.50,
    "gravity_cutoff_hz": 0.30,
    "lowpass_cutoff_hz": 20.0,
    "train_ratio": 0.70,
    "n_train_patterns": 7352,
    "n_test_patterns": 2947,
    # Table 4: "an overall accuracy of 96 % for the test data composed of 2947 patterns"
    "reported_accuracy": 0.96,
    "n_reported_correct": 2840,  # 2840/2947 = 96.37 % -> printed as 96 % in the paper
}

ACTIVITY_NAMES = {
    1: "Walking",
    2: "Walking upstairs",
    3: "Walking downstairs",
    4: "Sitting",
    5: "Standing",
    6: "Laying",
}
ACTIVITY_ABBREV = {1: "WK", 2: "WU", 3: "WD", 4: "ST", 5: "SD", 6: "LD"}
LABELS = tuple(sorted(ACTIVITY_NAMES))  # 1..6, the order used by Table 4

# Class counts of the official test set (test/y_test.txt).  They must equal the
# row sums of the confusion matrix of Table 4 - checked by the test-suite.
TEST_CLASS_COUNTS = np.array([496, 471, 420, 491, 532, 537])

# Confusion matrix of the paper's Table 4 (rows = actual class, columns =
# predicted class, order WK, WU, WD, ST, SD, LD).  In the PDF the numbers of a
# row are typeset without spaces and the text layer runs them together, so the
# matrix below is the unique reconstruction whose row sums are the exact test-set
# class counts and whose row/column percentages round to the recall and precision
# values printed by the authors (99/96/98/88/97/100 and 96/98/99/97/90/100).
PAPER_CONFUSION = np.array(
    [
        [492, 1, 3, 0, 0, 0],    # Walking          -> 496 patterns, recall  99 %
        [18, 451, 2, 0, 0, 0],   # Walking upstairs -> 471 patterns, recall  96 %
        [4, 6, 410, 0, 0, 0],    # Walking downst.  -> 420 patterns, recall  98 %
        [0, 2, 0, 432, 57, 0],   # Sitting          -> 491 patterns, recall  88 %
        [0, 0, 0, 14, 518, 0],   # Standing         -> 532 patterns, recall  97 %
        [0, 0, 0, 0, 0, 537],    # Laying           -> 537 patterns, recall 100 %
    ],
    dtype=int,
)
PAPER_RECALL_PCT = (99, 96, 98, 88, 97, 100)
PAPER_PRECISION_PCT = (96, 98, 99, 97, 90, 100)

DATASET_URLS = (
    "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip",
)
USER_AGENT = "Mozilla/5.0 (UCI-HAR-recreation)"

REQUIRED_FILES = (
    "features.txt",
    "activity_labels.txt",
    "train/X_train.txt",
    "train/y_train.txt",
    "train/subject_train.txt",
    "test/X_test.txt",
    "test/y_test.txt",
    "test/subject_test.txt",
)

# --------------------------------------------------------------------------- #
# 2. Dataset acquisition (cached download + (nested) zip extraction)
# --------------------------------------------------------------------------- #
def _dataset_root_present(base: Path) -> bool:
    return all((base / rel).is_file() for rel in REQUIRED_FILES)


def find_dataset_root(root: Path) -> Path | None:
    """Return the directory that holds features.txt / train / test, if any."""
    root = Path(root)
    if _dataset_root_present(root):
        return root
    for marker in root.rglob("features.txt"):
        candidate = marker.parent
        if _dataset_root_present(candidate):
            return candidate
    return None


def _download(url: str, dest: Path, timeout: int = 300) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response, open(dest, "wb") as handle:
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            handle.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r[data ] {dest.name}: {done / 1e6:7.1f}/{total / 1e6:.1f} MB "
                      f"({100.0 * done / total:5.1f} %)", end="", flush=True)
        print(flush=True)


def _extract(archive: Path, dest: Path) -> None:
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(dest)


def ensure_dataset(data_dir: Path, download: bool = True, url: str | None = None) -> Path:
    """Locate the UCI HAR dataset under *data_dir*, downloading it when missing.

    The UCI "static" endpoint serves an outer ZIP containing a second ZIP
    ("UCI HAR Dataset.zip"), hence the loop that also unpacks nested archives.
    """
    data_dir = Path(data_dir)
    found = find_dataset_root(data_dir)
    if found is not None:
        print(f"[data ] using cached dataset: {found}")
        return found

    if not download:
        raise FileNotFoundError(
            f"Dataset not found under {data_dir} and downloading is disabled. Download "
            "'UCI HAR Dataset.zip' from https://archive.ics.uci.edu/dataset/240/"
            f"human+activity+recognition+using+smartphones and unpack it into {data_dir}."
        )

    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir / "UCI_HAR_Dataset.zip"
    if not archive.is_file():
        urls = (url,) if url else DATASET_URLS
        last_error: Exception | None = None
        for candidate in urls:
            try:
                print(f"[data ] downloading dataset from {candidate}")
                _download(candidate, archive)
                break
            except Exception as exc:  # noqa: BLE001 - report and try the next mirror
                last_error = exc
                print(f"[data ] download failed ({type(exc).__name__}: {exc})")
        else:
            raise RuntimeError(f"Could not download the dataset: {last_error}")

    print(f"[data ] unpacking {archive.name} ...")
    _extract(archive, data_dir)

    for _ in range(3):  # nested archives (only needed for the UCI static endpoint)
        if find_dataset_root(data_dir) is not None:
            break
        nested = [p for p in data_dir.rglob("*.zip") if p != archive]
        if not nested:
            break
        for path in nested:
            print(f"[data ] unpacking nested {path.name} ...")
            _extract(path, path.parent)
            path.unlink(missing_ok=True)  # redundant once unpacked, keeps the cache small

    found = find_dataset_root(data_dir)
    if found is None:
        raise FileNotFoundError(f"Dataset files were not found under {data_dir} after unpacking.")
    print(f"[data ] dataset ready: {found}")
    return found


# --------------------------------------------------------------------------- #
# 3. Loading
# --------------------------------------------------------------------------- #
def load_split(base: Path, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load X (patterns x 561 features), y (activity 1..6) and subject ids."""
    folder = Path(base) / split
    X = np.loadtxt(folder / f"X_{split}.txt", dtype=np.float64)
    y = np.loadtxt(folder / f"y_{split}.txt", dtype=int)
    subjects = np.loadtxt(folder / f"subject_{split}.txt", dtype=int)
    return X, y, subjects


def load_feature_names(base: Path) -> list[str]:
    names = []
    for line in (Path(base) / "features.txt").read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip():
            names.append(line.split(maxsplit=1)[1].strip())
    return names


def load_activity_labels(base: Path) -> dict[int, str]:
    labels = {}
    for line in (Path(base) / "activity_labels.txt").read_text(encoding="utf-8").splitlines():
        if line.strip():
            index, name = line.split(maxsplit=1)
            labels[int(index)] = name.strip()
    return labels


# --------------------------------------------------------------------------- #
# 4. Statistical helpers (re-)implemented on top of numpy
# --------------------------------------------------------------------------- #
def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 6) -> np.ndarray:
    """Confusion matrix with rows = actual class, columns = predicted class."""
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    for actual, predicted in zip(np.asarray(y_true), np.asarray(y_pred)):
        matrix[int(actual) - 1, int(predicted) - 1] += 1
    return matrix


def recall_precision(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-class recall (sensitivity) and precision from a confusion matrix."""
    rows = matrix.sum(axis=1)
    cols = matrix.sum(axis=0)
    diagonal = np.diag(matrix).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        recall = np.where(rows > 0, diagonal / rows, 0.0)
        precision = np.where(cols > 0, diagonal / cols, 0.0)
    return recall, precision


def accuracy_from_matrix(matrix: np.ndarray) -> float:
    return float(np.trace(matrix)) / float(matrix.sum())


def per_subject_accuracy(subjects: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> list[tuple[int, int, float]]:
    """(subject id, number of test windows, accuracy) for every test subject."""
    rows = []
    for subject in sorted(set(int(s) for s in subjects)):
        mask = subjects == subject
        rows.append((subject, int(mask.sum()), float(np.mean(y_true[mask] == y_pred[mask]))))
    return rows


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval (95 % by default) for a binomial proportion."""
    if trials == 0:
        return 0.0, 1.0
    p = successes / trials
    denominator = 1.0 + z * z / trials
    centre = (p + z * z / (2.0 * trials)) / denominator
    half = z * np.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials)) / denominator
    return float(centre - half), float(centre + half)


# --------------------------------------------------------------------------- #
# 5. The study pipeline: multiclass SVM (One-Vs-All, RBF kernel, 10-fold CV)
# --------------------------------------------------------------------------- #
# Section 3 of the paper: "well-known and state-of-the-art Support Vector Machine
# (SVM) binary classifiers, which are generalized to the multiclass case through a
# One-Vs-All (OVA) approach: the SVM hyperparameters are selected through a
# 10-fold Cross Validation procedure and Gaussian kernels are used".
PAPER_GRID = {
    "estimator__C": [1.0, 10.0, 100.0, 1000.0],
    "estimator__gamma": ["scale", 1e-2, 1e-3],
}
FAST_GRID = {
    "estimator__C": [10.0, 100.0],
    "estimator__gamma": ["scale", 1e-2],
}
GRIDS = {"paper": PAPER_GRID, "fast": FAST_GRID}
QUICK_PARAMS = {"C": 100.0, "gamma": "scale"}


def make_estimator(C: float = 100.0, gamma: str | float = "scale", cache_mb: int = 500, jobs: int = 1):
    """Binary RBF SVM per class -> multiclass SVM with a One-Vs-All scheme."""
    return OneVsRestClassifier(
        SVC(kernel="rbf", C=C, gamma=gamma, cache_size=cache_mb, random_state=0),
        n_jobs=jobs,
    )


def make_folds(args) -> StratifiedKFold:
    """The CV splitter used for model selection.

    By default the folds are contiguous blocks of the recordings (sklearn's
    default for integer `cv`), which is the pessimistic and realistic setting:
    neighbouring windows overlap by 50 % and are almost duplicates, so shuffling
    them into different folds inflates the CV accuracy.  `--shuffle-cv` enables
    the optimistic variant.
    """
    return StratifiedKFold(n_splits=args.cv, shuffle=args.shuffle_cv,
                           random_state=args.seed if args.shuffle_cv else None)


def select_hyperparameters(
    X: np.ndarray,
    y: np.ndarray,
    folds: StratifiedKFold,
    jobs: int = -1,
    grid: dict | None = None,
) -> tuple[list[dict], dict, float]:
    """CV over (C, gamma) of the RBF kernel - the paper's model selection.

    Returns (all records sorted by CV accuracy, best record, seconds elapsed).
    Progress is printed per setting so long runs stay transparent.
    """
    grid = grid if grid is not None else PAPER_GRID
    records: list[dict] = []
    started = time.time()
    for c_value in grid["estimator__C"]:
        for gamma in grid["estimator__gamma"]:
            scores = cross_val_score(make_estimator(c_value, gamma, jobs=1), X, y,
                                     cv=folds, n_jobs=jobs, scoring="accuracy")
            record = {"C": float(c_value), "gamma": gamma,
                      "mean": float(scores.mean()), "std": float(scores.std())}
            records.append(record)
            print(f"[train]   C={record['C']:<7g} gamma={str(gamma):<6} -> "
                  f"CV accuracy {record['mean']:.4f} +/- {record['std']:.4f}", flush=True)
    best = max(records, key=lambda record: record["mean"])  # argmax = the paper's selection rule
    return sorted(records, key=lambda record: -record["mean"]), best, time.time() - started


# --------------------------------------------------------------------------- #
# 6. Tiny test-suite / reporting helpers
# --------------------------------------------------------------------------- #
@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""


def print_checks(checks: list[Check]) -> bool:
    width = max([len(c.name) for c in checks] + [46])
    print("\n" + "=" * 78)
    print("TEST REPORT".center(78))
    print("=" * 78)
    for check in checks:
        status = "PASS" if check.passed else "FAIL"
        print(f"[{status}] {check.name:<{width}} {check.detail}")
    failed = [c for c in checks if not c.passed]
    print("-" * 78)
    print(f"{len(checks) - len(failed)}/{len(checks)} checks passed"
          + ("" if not failed else f"  -> FAILED: {', '.join(c.name for c in failed)}"))
    return not failed


def table_layout(row_labels: list[str], col_labels: list[str],
                 stat_labels: tuple[str, ...] = ("recall", "precision", "overall accuracy"),
                 minimum_cell: int = 7) -> tuple[int, int]:
    """Column widths so that the confusion matrices and their % rows line up."""
    header_width = max([len(label) for label in row_labels] + [len(label) for label in stat_labels])
    cell_width = max([len(label) for label in col_labels] + [minimum_cell])
    return header_width, cell_width


def fmt_matrix(matrix: np.ndarray, row_labels: list[str], col_labels: list[str],
               layout: tuple[int, int]) -> str:
    header_width, cell_width = layout
    lines = [" " * header_width + "".join(f"{label:>{cell_width}}" for label in col_labels)]
    for label, row in zip(row_labels, matrix):
        lines.append(f"{label:>{header_width}}" + "".join(f"{int(v):>{cell_width}d}" for v in row))
    return "\n".join(lines)


def fmt_signed_matrix(matrix: np.ndarray, row_labels: list[str], col_labels: list[str],
                      layout: tuple[int, int]) -> str:
    header_width, cell_width = layout
    lines = [" " * header_width + "".join(f"{label:>{cell_width}}" for label in col_labels)]
    for label, row in zip(row_labels, matrix):
        lines.append(f"{label:>{header_width}}" + "".join(f"{int(v):>+{cell_width}d}" for v in row))
    return "\n".join(lines)


def fmt_pct_row(name: str, values: np.ndarray, layout: tuple[int, int]) -> str:
    header_width, cell_width = layout
    return f"{name:>{header_width}}" + "".join(f"{100.0 * v:>{cell_width - 1}.0f}%" for v in values)


def print_paper_table() -> None:
    rows = [ACTIVITY_ABBREV[label] for label in LABELS]
    row_labels = [f"{ACTIVITY_NAMES[label]} ({ACTIVITY_ABBREV[label]})" for label in LABELS]
    layout = table_layout(row_labels, rows)
    recall, precision = recall_precision(PAPER_CONFUSION)
    print("\n--- paper, Table 4 (multiclass SVM, 2947 test patterns) ---")
    print(fmt_matrix(PAPER_CONFUSION, row_labels, rows, layout))
    print(fmt_pct_row("recall", recall, layout))
    print(fmt_pct_row("precision", precision, layout))
    print(fmt_pct_row("overall accuracy", np.array([accuracy_from_matrix(PAPER_CONFUSION)]), layout)
          + f"   = {np.trace(PAPER_CONFUSION)}/{PAPER_CONFUSION.sum()} correct, published as 96 %")


def print_cv_table(records: list[dict], cv: int = 10, top: int = 6) -> None:
    """Summary of the CV search, best settings first."""
    shown = records[:top]
    print(f"\n--- {cv}-fold cross-validation results (best {len(shown)} of {len(records)} settings) ---")
    print(f"{'C':>8}{'gamma':>12}{'CV accuracy':>14}{'std':>9}")
    for record in shown:
        gamma = record["gamma"] if isinstance(record["gamma"], str) else f"{record['gamma']:g}"
        print(f"{record['C']:>8g}{gamma:>12}{record['mean']:>14.4f}{record['std']:>9.4f}")


# --------------------------------------------------------------------------- #
# 7. The checks (this is the "tests its accuracy" part)
# --------------------------------------------------------------------------- #
EXPECTED_TRAIN_CLASS_COUNTS = np.array([1226, 1073, 986, 1286, 1374, 1407])
EXPECTED_ACTIVITY_LABELS = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING",
}
EXPECTED_TEST_SUBJECTS = (2, 4, 9, 10, 12, 13, 18, 20, 24)


def dataset_checks(
    X_train: np.ndarray,
    y_train: np.ndarray,
    subj_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    subj_test: np.ndarray,
    feature_names: list[str],
    activity_labels: dict[int, str],
) -> list[Check]:
    """Validate the downloaded data against the facts published in the paper/README."""
    checks: list[Check] = []

    ok = X_train.shape == (PAPER["n_train_patterns"], PAPER["n_features"]) and X_test.shape == (
        PAPER["n_test_patterns"], PAPER["n_features"])
    checks.append(Check("dataset layout 7352x561 / 2947x561", ok,
                        f"X_train {X_train.shape}, X_test {X_test.shape}"))

    checks.append(Check("561 named features (features.txt)", len(feature_names) == PAPER["n_features"],
                        f"{len(feature_names)} names, first = '{feature_names[0] if feature_names else '-'}'"))

    labels_ok = activity_labels == EXPECTED_ACTIVITY_LABELS
    checks.append(Check("6 activities, README label order", labels_ok,
                        ", ".join(f"{k}={v}" for k, v in sorted(activity_labels.items()))))

    finite = bool(np.isfinite(X_train).all() and np.isfinite(X_test).all())
    in_range = bool(X_train.min() >= -1.0 - 1e-9 and X_train.max() <= 1.0 + 1e-9
                    and X_test.min() >= -1.0 - 1e-9 and X_test.max() <= 1.0 + 1e-9)
    checks.append(Check("features finite and normalised in [-1,1]", finite and in_range,
                        f"train [{X_train.min():.3f}, {X_train.max():.3f}], test [{X_test.min():.3f}, {X_test.max():.3f}], "
                        f"no NaN/inf = {finite}"))

    train_counts = np.bincount(y_train, minlength=7)[1:]
    test_counts = np.bincount(y_test, minlength=7)[1:]
    counts_ok = bool(np.array_equal(train_counts, EXPECTED_TRAIN_CLASS_COUNTS)
                     and np.array_equal(test_counts, TEST_CLASS_COUNTS)
                     and len(y_train) + len(y_test) == 10299)
    checks.append(Check("class distribution matches the dataset", counts_ok,
                        f"train {train_counts.tolist()}, test {test_counts.tolist()}"))

    train_subjects = set(int(s) for s in subj_train)
    test_subjects = set(int(s) for s in subj_test)
    split_ok = (train_subjects.isdisjoint(test_subjects)
                and train_subjects | test_subjects == set(range(1, PAPER["n_subjects"] + 1))
                and len(train_subjects) == 21 and len(test_subjects) == 9)
    checks.append(Check("70 % / 30 % split by subject", split_ok,
                        f"{len(train_subjects)} train subjects / {len(test_subjects)} test subjects "
                        f"({len(train_subjects) / PAPER['n_subjects']:.0%} of the 30 volunteers)"))

    checks.append(Check("test subjects are the published 9", tuple(sorted(test_subjects)) == EXPECTED_TEST_SUBJECTS,
                        f"test subject ids {sorted(test_subjects)}"))

    window_ok = (abs(PAPER["window_seconds"] * PAPER["sampling_rate_hz"] - PAPER["window_samples"]) < 1e-9
                 and abs(PAPER["overlap"] - 0.5) < 1e-9)
    checks.append(Check("window geometry 2.56 s @ 50 Hz = 128", window_ok,
                        f"{PAPER['window_seconds']} s x {PAPER['sampling_rate_hz']} Hz = "
                        f"{PAPER['window_seconds'] * PAPER['sampling_rate_hz']:.0f} samples, "
                        f"overlap {PAPER['overlap']:.0%}"))
    return checks


def paper_table_checks() -> list[Check]:
    """Self-consistency of the reconstructed Table 4 (paper's own numbers)."""
    checks: list[Check] = []
    row_sums = PAPER_CONFUSION.sum(axis=1)
    checks.append(Check("Table 4 row sums = test class counts", bool(np.array_equal(row_sums, TEST_CLASS_COUNTS)),
                        f"rows {row_sums.tolist()} == y_test counts {TEST_CLASS_COUNTS.tolist()}"))

    recall, precision = recall_precision(PAPER_CONFUSION)
    recall_ok = tuple(int(round(100 * v)) for v in recall) == PAPER_RECALL_PCT
    precision_ok = tuple(int(round(100 * v)) for v in precision) == PAPER_PRECISION_PCT
    checks.append(Check("Table 4 reproduces published recall/precision", recall_ok and precision_ok,
                        f"recall {tuple(int(round(100 * v)) for v in recall)} / "
                        f"precision {tuple(int(round(100 * v)) for v in precision)}"))

    correct = int(np.trace(PAPER_CONFUSION))
    accuracy = accuracy_from_matrix(PAPER_CONFUSION)
    checks.append(Check("Table 4 gives the published 96 % accuracy",
                        correct == PAPER["n_reported_correct"] and round(100 * accuracy) == 96,
                        f"{correct}/{PAPER_CONFUSION.sum()} = {100 * accuracy:.2f} % -> 96 % as published"))
    return checks


def fitted_gamma(binary_svm, X: np.ndarray) -> float:
    """Effective numeric gamma of a fitted SVC, for any gamma setting/version."""
    for attribute in ("gamma_", "_gamma"):
        value = getattr(binary_svm, attribute, None)
        if value is not None:
            return float(value)
    gamma = binary_svm.gamma
    if isinstance(gamma, str):  # 'scale' -> 1 / (n_features * var(X)), 'auto' -> 1 / n_features
        n_features = X.shape[1]
        if gamma == "scale":
            variance = float(X.var())
            return 1.0 / (n_features * variance) if variance > 0 else 1.0
        return 1.0 / n_features
    return float(gamma)


def train_model(args, X_train: np.ndarray, y_train: np.ndarray):
    """Train the paper's classifier. Returns (estimator, info)."""
    folds = make_folds(args)
    info: dict = {"cv_records": None, "cv_score": float("nan"), "fit_seconds": 0.0}

    if args.quick:
        params = dict(QUICK_PARAMS)
        print(f"\n[train] quick mode: single hyper-parameter set C={params['C']:g}, gamma={params['gamma']}")
        if not args.no_cv:
            print(f"[train] {args.cv}-fold cross-validation of that setting ...", flush=True)
            scores = cross_val_score(make_estimator(params["C"], params["gamma"], jobs=1), X_train, y_train,
                                     cv=folds, n_jobs=args.jobs, scoring="accuracy")
            info["cv_score"] = float(scores.mean())
            print(f"[train] CV accuracy = {scores.mean():.4f} +/- {scores.std():.4f}")
        estimator = make_estimator(params["C"], params["gamma"], cache_mb=args.cache_mb, jobs=args.jobs)
    else:
        grid = GRIDS[args.grid]
        settings = len(grid["estimator__C"]) * len(grid["estimator__gamma"])
        print(f"\n[train] selecting C and gamma with {args.cv}-fold cross-validation "
              f"({settings} settings = {settings * args.cv} SVM trainings, this can take a few minutes; "
              f"--quick is much faster) ...", flush=True)
        records, best, elapsed = select_hyperparameters(X_train, y_train, folds,
                                                        jobs=args.jobs, grid=grid)
        info["cv_records"] = records
        info["cv_score"] = float(best["mean"])
        info["fit_seconds"] += elapsed
        print_cv_table(records, cv=args.cv)
        print(f"[train] selected C={best['C']:g}, gamma={best['gamma']} "
              f"(best CV accuracy {best['mean']:.4f})", flush=True)
        estimator = make_estimator(best["C"], best["gamma"], cache_mb=args.cache_mb, jobs=args.jobs)

    started = time.time()
    estimator.fit(X_train, y_train)
    info["fit_seconds"] += time.time() - started
    info["params"] = {
        "C": float(QUICK_PARAMS["C"]),
        "gamma": QUICK_PARAMS["gamma"],
        "gamma_effective": float("nan"),
    }
    if hasattr(estimator, "estimators_") and estimator.estimators_:
        binary = estimator.estimators_[0]
        info["params"] = {
            "C": float(binary.C),
            "gamma": binary.gamma,
            "gamma_effective": fitted_gamma(binary, X_train),
        }
    info["n_support_vectors"] = int(sum(len(est.support_) for est in estimator.estimators_)) \
        if getattr(estimator, "estimators_", None) else 0
    return estimator, info


def result_checks(
    accuracy: float,
    matrix: np.ndarray,
    recall: np.ndarray,
    precision: np.ndarray,
    info: dict,
    args,
) -> list[Check]:
    """Compare the reproduced run with the numbers published in the paper."""
    checks: list[Check] = []
    paper_accuracy = PAPER["n_reported_correct"] / PAPER["n_test_patterns"]  # 0.96369 -> "96 %"
    correct = int(np.trace(matrix))

    delta = accuracy - paper_accuracy
    checks.append(Check("overall accuracy reproduces the paper's 96 %",
                        abs(delta) <= args.tolerance,
                        f"{100 * accuracy:.2f} % vs {100 * paper_accuracy:.2f} % "
                        f"({correct}/{matrix.sum()} correct, delta {100 * delta:+.2f} pp, "
                        f"tolerance {100 * args.tolerance:.0f} pp)"))

    low, high = wilson_interval(correct, int(matrix.sum()))
    checks.append(Check("paper's value inside our 95 % confidence interval",
                        low <= paper_accuracy <= high,
                        f"95 % Wilson CI [{100 * low:.2f} %, {100 * high:.2f} %] contains {100 * paper_accuracy:.2f} %"))

    reference = 0.908  # paper cites 90.8 % (Karantonis et al.) for waist-worn sensors
    checks.append(Check("at least as good as previous-work baseline", accuracy >= reference,
                        f"{100 * accuracy:.2f} % >= 90.8 % (accuracy reported for dedicated sensors in the paper)"))

    recall_dev = np.abs(recall - np.array(PAPER_RECALL_PCT) / 100.0)
    precision_dev = np.abs(precision - np.array(PAPER_PRECISION_PCT) / 100.0)
    worst_class = int(np.argmax(np.maximum(recall_dev, precision_dev)))
    checks.append(Check("per-class recall/precision within tolerance",
                        max(recall_dev.max(), precision_dev.max()) <= args.class_tolerance,
                        f"largest deviation {100 * max(recall_dev.max(), precision_dev.max()):.1f} pp on "
                        f"{ACTIVITY_NAMES[LABELS[worst_class]]} (recall {100 * recall_dev[worst_class]:.1f} pp, "
                        f"precision {100 * precision_dev[worst_class]:.1f} pp, "
                        f"tolerance {100 * args.class_tolerance:.0f} pp)"))

    pair_scores = {
        frozenset((a, b)): int(matrix[a - 1, b - 1] + matrix[b - 1, a - 1])
        for i, a in enumerate(LABELS) for b in LABELS[i + 1:]
    }
    most_confused = max(pair_scores, key=lambda pair: pair_scores[pair])
    checks.append(Check("Sitting/Standing are the hardest pair (as reported)",
                        most_confused == frozenset((4, 5)),
                        f"most confused pair = {'/'.join(ACTIVITY_NAMES[l] for l in sorted(most_confused))} "
                        f"({pair_scores[most_confused]} windows)"))

    if np.isfinite(info["cv_score"]):
        checks.append(Check("test accuracy is in line with the CV estimate",
                            abs(accuracy - info["cv_score"]) <= args.class_tolerance,
                            f"{args.cv}-fold CV {100 * info['cv_score']:.2f} % vs test {100 * accuracy:.2f} %"))

    if info["cv_records"]:
        best_record = info["cv_records"][0]  # records are sorted best first
        chosen = (float(info["params"]["C"]), str(info["params"]["gamma"]))
        selected = (float(best_record["C"]), str(best_record["gamma"]))
        checks.append(Check("hyper-parameters = CV argmax (paper's rule)", chosen == selected,
                            f"C={chosen[0]:g}, gamma={chosen[1]} won with "
                            f"{100 * best_record['mean']:.2f} % among {len(info['cv_records'])} CV settings"))
    return checks


def determinism_check(estimator, X_train: np.ndarray, y_train: np.ndarray,
                      X_test: np.ndarray, y_pred: np.ndarray) -> Check:
    """Refit an identical model: the paper's protocol must be reproducible."""
    refit = clone(estimator)
    refit.fit(X_train, y_train)
    again = refit.predict(X_test)
    identical = bool(np.array_equal(again, y_pred))
    return Check("model refit is reproducible (identical predictions)", identical,
                 f"{'identical' if identical else 'different'} predictions from a second identical fit "
                 f"({int((again != y_pred).sum())} differing windows)")


# --------------------------------------------------------------------------- #
# 8. Command line + orchestration
# --------------------------------------------------------------------------- #
def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="har_mcsvm_recreation.py",
        description="Recreate the multiclass-SVM results of the UCI HAR study "
                    "(Anguita et al., ESANN 2013) and test the reproduced accuracy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="Exit code: 0 = all checks passed, 1 = a check failed, 2 = data/setup error.",
    )
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent / "data",
                        help="folder that holds (or receives) the UCI HAR dataset")
    parser.add_argument("--dataset-url", default=None, help="override the download URL")
    parser.add_argument("--no-download", action="store_true", help="never access the network")
    parser.add_argument("--quick", action="store_true",
                        help="skip the 10-fold CV grid search and use a single (C, gamma) pair (~15 s)")
    parser.add_argument("--no-cv", action="store_true", help="skip the CV estimate in --quick mode as well")
    parser.add_argument("--grid", choices=sorted(GRIDS), default="paper",
                        help="hyper-parameter grid for the CV search: 'paper' is exhaustive, 'fast' is smaller")
    parser.add_argument("--cv", type=int, default=10, help="number of CV folds for model selection")
    parser.add_argument("--shuffle-cv", action="store_true",
                        help="shuffle the CV folds (optimistic: 50 %% overlapping windows become almost "
                             "duplicates in different folds); off by default")
    parser.add_argument("--jobs", type=int, default=-1, help="parallel jobs (-1 = all cores)")
    parser.add_argument("--seed", type=int, default=0, help="random seed when --shuffle-cv is used")
    parser.add_argument("--cache-mb", type=int, default=500, help="libsvm kernel cache per binary SVM (MB)")
    parser.add_argument("--tolerance", type=float, default=0.02,
                        help="allowed |accuracy - 96.37 %%| for the reproduction check")
    parser.add_argument("--class-tolerance", type=float, default=0.05,
                        help="allowed per-class deviation from the paper's Table 4")
    parser.add_argument("--list-checks", action="store_true",
                        help="only run the offline checks (dataset + paper table), skip training")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    started = time.time()

    print("=" * 78)
    print("UCI HAR RECREATION - MULTICLASS SVM (ONE-VS-ALL, RBF KERNEL)".center(78))
    print("=" * 78)
    print(f"paper    : {PAPER['title']}")
    print(f"           {PAPER['authors']}")
    print(f"           {PAPER['venue']}")
    print(f"dataset  : {PAPER['dataset']}")
    print(f"protocol : {PAPER['n_subjects']} subjects, {PAPER['n_features']} features, "
          f"{PAPER['n_train_patterns']} train / {PAPER['n_test_patterns']} test patterns, "
          f"{PAPER['sampling_rate_hz']} Hz, {PAPER['window_seconds']} s windows "
          f"({PAPER['overlap']:.0%} overlap)")
    print(f"model    : multiclass SVM = OneVsAll of binary RBF-kernel SVMs; "
          f"{args.cv}-fold CV model selection")
    print(f"mode     : " + ("quick, single (C, gamma) pair" if args.quick
                            else f"grid search over C and gamma with {args.cv}-fold CV (grid '{args.grid}')"))

    try:
        base = ensure_dataset(args.data_dir, download=not args.no_download, url=args.dataset_url)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"\n[data ] ERROR: {exc}", file=sys.stderr)
        return 2

    X_train, y_train, subj_train = load_split(base, "train")
    X_test, y_test, subj_test = load_split(base, "test")
    feature_names = load_feature_names(base)
    activity_labels = load_activity_labels(base)
    print(f"[data ] loaded {X_train.shape[0]} training and {X_test.shape[0]} test patterns "
          f"with {X_train.shape[1]} features")

    checks = dataset_checks(X_train, y_train, subj_train, X_test, y_test, subj_test,
                            feature_names, activity_labels)
    checks += paper_table_checks()
    print_paper_table()

    if args.list_checks:
        all_passed = print_checks(checks)
        return 0 if all_passed else 1

    estimator, info = train_model(args, X_train, y_train)
    print(f"[train] trained OneVsAll(RBF SVM): C={info['params']['C']:g}, "
          f"gamma={info['params']['gamma']} (effective {info['params']['gamma_effective']:.5f}), "
          f"{info['n_support_vectors']} support vectors, {info['fit_seconds']:.1f} s")

    y_pred = estimator.predict(X_test)
    matrix = confusion_matrix(y_test, y_pred)
    recall, precision = recall_precision(matrix)
    accuracy = accuracy_from_matrix(matrix)

    row_labels = [f"{ACTIVITY_NAMES[label]} ({ACTIVITY_ABBREV[label]})" for label in LABELS]
    column_labels = [ACTIVITY_ABBREV[label] for label in LABELS]
    layout = table_layout(row_labels, column_labels)
    print("\n--- reproduced confusion matrix (this run, 2947 test patterns) ---")
    print(fmt_matrix(matrix, row_labels, column_labels, layout))
    print(fmt_pct_row("recall", recall, layout))
    print(fmt_pct_row("precision", precision, layout))
    print(fmt_pct_row("overall accuracy", np.array([accuracy]), layout)
          + f"   = {int(np.trace(matrix))}/{int(matrix.sum())} correct "
            f"(paper: 96 %, {PAPER['n_reported_correct']}/{PAPER['n_test_patterns']})")

    difference = matrix - PAPER_CONFUSION
    print("\n--- difference to the paper's Table 4 (this run - paper) ---")
    print(fmt_signed_matrix(difference, row_labels, column_labels, layout))
    print(f"identical cells {int((difference == 0).sum())}/{matrix.size}, "
          f"total absolute deviation {int(np.abs(difference).sum())} windows, "
          f"max cell deviation {int(np.abs(difference).max())}")

    print("\n--- per-subject accuracy on the 9 test subjects ---")
    for subject, count, subject_accuracy in per_subject_accuracy(subj_test, y_test, y_pred):
        print(f"subject {subject:>2}: {count:>4} windows, accuracy {100 * subject_accuracy:>6.2f} %")

    checks.append(determinism_check(estimator, X_train, y_train, X_test, y_pred))
    checks += result_checks(accuracy, matrix, recall, precision, info, args)

    all_passed = print_checks(checks)
    print(f"\ntotal runtime: {time.time() - started:.1f} s")
    if all_passed:
        print("RESULT: the multiclass SVM reproduces the paper's 96 % on the official test set.")
    else:
        print("RESULT: at least one check failed - see the report above.")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
