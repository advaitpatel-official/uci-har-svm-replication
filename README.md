# UCI HAR Recreation — Multiclass SVM (One-Vs-All, RBF kernel)

**Short description**

A single-file, self-contained recreation of the UCI *Human Activity Recognition Using
Smartphones* study (Anguita et al., ESANN 2013). It downloads and validates the official
561-feature dataset, trains the paper's classifier — a multiclass SVM built as six
One-Vs-All binary RBF-kernel SVMs whose `C` and `gamma` are selected by 10-fold
cross-validation — then evaluates it on the official 2947-pattern test set (9 subjects
never seen during training) and reports the reproduced accuracy, confusion matrix and
per-class recall/precision side by side with the numbers published in the paper.

**Reproduced result: 2855/2947 = 96.88 % test accuracy, against the paper's
2840/2947 = 96.37 % (printed as "96 %"). All self-checks pass.**

---

## The study

| Item | Detail |
| --- | --- |
| Paper | D. Anguita, A. Ghio, L. Oneto, X. Parra, J. L. Reyes-Ortiz, *A Public Domain Dataset for Human Activity Recognition Using Smartphones*, ESANN 2013, Bruges (Belgium), 24-26 April 2013, pp. 437-442 |
| Paper PDF | <https://www.esann.org/sites/default/files/proceedings/legacy/es2013-84.pdf> |
| Dataset | UCI Machine Learning Repository, id 240 — <https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones> |
| Dataset DOI | `10.24432/C54S4K` (CC BY 4.0), 10 299 instances, 561 features |
| Download URLs | `https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip` (mirror: `.../ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip`) |
| Companion paper (classifier family) | Anguita et al., *Human Activity Recognition on Smartphones using a Multiclass Hardware-Friendly Support Vector Machine*, IWAAL 2012, LNCS 7657, pp. 216-223, DOI `10.1007/978-3-642-35395-6_30` |
| Companion paper (fixed-point / energy) | Anguita et al., *Energy Efficient Smartphone-Based Activity Recognition using Fixed-Point Arithmetic*, JUCS 19(9), May 2013 |

### The protocol being reproduced

* 30 volunteers (19-48 years), 6 activities: WALKING, WALKING_UPSTAIRS,
  WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING.
* Samsung Galaxy S II worn on the waist; accelerometer + gyroscope at 50 Hz.
* Noise filtering (median filter + 3rd-order Butterworth low-pass, 20 Hz corner);
  gravity separated from body acceleration with a 0.3 Hz Butterworth low-pass.
* Sliding windows of 2.56 s = 128 samples with 50 % overlap; 561 features per window
  from 17 time/frequency signals; features normalised and bounded within [-1, 1].
* Split 70 % / 30 % **by subject**: 21 subjects -> 7352 training patterns,
  9 subjects -> 2947 test patterns.
* Classifier: multiclass SVM = One-Vs-All of binary Gaussian-kernel SVMs,
  hyper-parameters chosen by 10-fold cross-validation.
* Published result: **96 % overall accuracy** on the 2947 test patterns.

---

## The algorithm

```text
X (7352 x 561)  ->  OneVsRestClassifier( SVC(kernel="rbf", C, gamma) )  ->  y (1..6)
                       |                                                     ^
                       +-- 6 binary "class k vs. rest" SVMs ---- argmax_k f_k(x)
```

Each binary SVM solves the soft-margin C-SVC problem
`min ½‖w‖² + C·Σξᵢ  s.t.  yᵢ(w·φ(xᵢ)+b) ≥ 1 − ξᵢ, ξᵢ ≥ 0`
with the RBF kernel `K(x, x′) = exp(−γ‖x − x′‖²)`. The kernel makes the decision
boundary non-linear in the 561-dimensional input space, which is what separates the
walking variants from each other and sitting from standing.

* `C` (`estimator__C`) — margin-violation penalty.
* `gamma` (`estimator__gamma`) — kernel width; `"scale"` means
  `1 / (n_features · Var(X))`, which evaluates to `0.00632` for these 561 features.
* Prediction = `argmax` over the 6 per-class decision values (One-Vs-All rule).
* Model selection = the paper's rule: **argmax of the mean 10-fold CV accuracy** over
  `C ∈ {1, 10, 100, 1000} × gamma ∈ {scale, 1e-2, 1e-3}` (12 settings, 120 SVM trainings).

The CV folds are **contiguous blocks of the recordings** by default (not shuffled),
because neighbouring windows overlap by 50 % and are near-duplicates: shuffling them
across folds would leak near-copies and inflate the CV score. `--shuffle-cv` enables
the optimistic variant.

---

## Repository layout

```text
UCI_HAR_Recreation/
├── har_mcsvm_recreation.py     # the whole study: data handling, classifier, accuracy, tests
├── README.md                   # this file
└── data/
    ├── UCI_HAR_Dataset.zip     # cached official download (~58 MB)
    └── UCI HAR Dataset/        # unpacked dataset
        ├── README.txt          # official dataset description
        ├── features.txt        # 561 feature names
        ├── activity_labels.txt # class index -> activity name
        ├── train/              # X_train.txt (7352x561), y_train.txt, subject_train.txt,
        │                       #   Inertial Signals/ (raw windows, unused here)
        └── test/               # X_test.txt (2947x561), y_test.txt, subject_test.txt,
                                #   Inertial Signals/ (raw windows, unused here)
```

## Requirements

The script bootstraps its own dependencies: if a package is missing it runs
`pip install` for it before importing. Nothing else is needed.

* Python 3.10+ (verified on 3.14.6)
* numpy (verified 2.5.3), scipy (verified 1.18.1), scikit-learn (verified 1.9.1)

If the dataset is not present under `--data-dir`, it is downloaded and unpacked
automatically (the UCI "static" endpoint serves a ZIP inside a ZIP; nested archives are
handled), otherwise the cached copy in `data/` is used.

---

## Usage

```text
python har_mcsvm_recreation.py               # full study protocol: 10-fold CV grid search
                                             #   over 12 (C, gamma) settings (~4-15 min, multicore)
python har_mcsvm_recreation.py --grid fast   # smaller grid (~1/3 of the cost)
python har_mcsvm_recreation.py --quick       # single (C, gamma) set (~30 s)
python har_mcsvm_recreation.py --list-checks # offline checks only, no training
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--data-dir` | `./data` | folder that holds (or receives) the UCI HAR dataset |
| `--dataset-url` | — | override the download URL |
| `--no-download` | off | never access the network |
| `--quick` | off | skip the CV grid search, use a single `(C, gamma)` pair |
| `--no-cv` | off | also skip the CV estimate in `--quick` mode |
| `--grid` | `paper` | hyper-parameter grid: `paper` (exhaustive) or `fast` |
| `--cv` | `10` | number of CV folds for model selection |
| `--shuffle-cv` | off | shuffle the CV folds (optimistic; see above) |
| `--jobs` | `-1` | parallel jobs (`-1` = all cores) |
| `--seed` | `0` | random seed when `--shuffle-cv` is used |
| `--cache-mb` | `500` | libsvm kernel cache per binary SVM (MB) |
| `--tolerance` | `0.02` | allowed `\|accuracy − 96.37 %\|` for the reproduction check |
| `--class-tolerance` | `0.05` | allowed per-class deviation from the paper's Table 4 |
| `--list-checks` | off | run only the offline checks (dataset + paper table) |

**Exit codes:** `0` = all checks passed, `1` = a check failed, `2` = data/setup error.


---

## Example run (`--quick`)

```text
[train] quick mode: single hyper-parameter set C=100, gamma=scale
[train] 10-fold cross-validation of that setting ...
[train] CV accuracy = 0.9569 +/- 0.0398
[train] trained OneVsAll(RBF SVM): C=100, gamma=scale (effective 0.00632), 1592 support vectors, 4.9 s

--- reproduced confusion matrix (this run, 2947 test patterns) ---
                            WK     WU     WD     ST     SD     LD
           Walking (WK)    489      7      0      0      0      0
  Walking upstairs (WU)     11    460      0      0      0      0
Walking downstairs (WD)      4     11    405      0      0      0
           Sitting (ST)      0      3      0    441     46      1
          Standing (SD)      1      0      0      8    523      0
            Laying (LD)      0      0      0      0      0    537
                 recall    99%    98%    96%    90%    98%   100%
              precision    97%    96%   100%    98%    92%   100%
       overall accuracy    97%   = 2855/2947 correct (paper: 96 %, 2840/2947)

18/18 checks passed
total runtime: 28.5 s
RESULT: the multiclass SVM reproduces the paper's 96 % on the official test set.
```

With the CV grid search (`--grid fast`) the selection follows the paper's rule and lands
on the same setting: `C=100, gamma=scale` (best CV accuracy 95.69 %), 19/19 checks pass.

---

## How the accuracy is calculated

Every number is re-derived from the 2947 predicted labels; no library summary metric is
trusted.

| Quantity | Definition in code | Reproduced value |
| --- | --- | --- |
| Confusion matrix `M` | `confusion_matrix()`, rows = actual, columns = predicted | compared cell-by-cell with the paper's Table 4 |
| Overall accuracy | `trace(M) / sum(M)` = correct / total (micro-average) | **2855/2947 = 96.88 %** (paper 2840/2947 = 96.37 %) |
| Recall per class | `M[k,k] / sum_j M[k,j]` (macro) | 99 / 98 / 96 / 90 / 98 / 100 % |
| Precision per class | `M[k,k] / sum_i M[i,k]` (macro) | 97 / 96 / 100 / 98 / 92 / 100 % |
| CV accuracy | mean of the 10 per-fold accuracies from `cross_val_score(scoring="accuracy")` | 95.69 % +/- 3.98 % |
| Confidence interval | 95 % Wilson score interval for a binomial proportion | [96.19 %, 97.45 %] contains the paper's 96.37 % |
| Per-subject accuracy | `mean(y_true == y_pred)` per test subject | 92.36 % - 99.74 % |
| Agreement with the paper | signed cell-by-cell `M - PAPER_CONFUSION` | 21/36 identical cells, 74 windows total deviation |

### The self-checks that decide pass/fail

Offline (also runnable via `--list-checks`, 11 checks):

1. dataset layout `7352x561` / `2947x561`;
2. 561 named features (`features.txt`);
3. 6 activities in README label order;
4. all features finite and normalised within `[-1, 1]`;
5. class distribution matches the dataset (`train [1226, 1073, 986, 1286, 1374, 1407]`,
   `test [496, 471, 420, 491, 532, 537]`, 10 299 patterns in total);
6. 70 % / 30 % split by subject (21 vs 9 subjects, disjoint);
7. the test subjects are the published `2, 4, 9, 10, 12, 13, 18, 20, 24`;
8. window geometry 2.56 s x 50 Hz = 128 samples with 50 % overlap;
9. the encoded Table 4 has row sums equal to the official test class counts;
10. Table 4 reproduces the published recall `(99, 96, 98, 88, 97, 100)` and
    precision `(96, 98, 99, 97, 90, 100)`;
11. Table 4 yields the published 96 % accuracy (2840/2947).

After training, more checks are added:

12. a refit of the identical model produces bit-identical predictions (determinism);
13. overall accuracy reproduces the paper's 96 % within `--tolerance` (default 2 pp);
14. the paper's value lies inside our 95 % Wilson confidence interval;
15. accuracy is at least the 90.8 % previous-work baseline cited by the paper;
16. per-class recall/precision are within `--class-tolerance` of Table 4;
17. Sitting/Standing is the most confused pair (54 windows here), as reported;
18. test accuracy is in line with the CV estimate (96.88 % vs 95.69 %);
19. the hyper-parameters used are the CV argmax (the paper's selection rule) - grid mode only.

---

## Notes and limitations

* **Classification stage only.** The script consumes the published, already windowed and
  normalised 561-feature files (`X_train.txt` / `X_test.txt`). It does not re-implement
  (and cannot verify) the median/Butterworth filtering, gravity separation, windowing or
  feature extraction; the `Inertial Signals/` raw windows are present in the download but
  unused. The window geometry is asserted only as a documentation/consistency check.
* **The split is by subject**, so the test set is 9 unseen people and 96.88 % measures
  generalisation to new users, not to new windows of known users.
* **Model selection happens inside the training set only** (10-fold CV), so the reported
  test accuracy is an unbiased estimate; the script flags any gap larger than 5 pp
  between the CV and test figures.
* **Small deviations from the paper are expected.** This run lands at 96.88 % vs the
  published 96.37 %, and 15 of 36 confusion-matrix cells differ. Sources: the authors'
  own SVM used different arithmetic/implementation details (their follow-up work is a
  fixed-point SVM), and this project reconstructs the printed Table 4 from the PDF text
  layer - constrained to the unique matrix whose row sums equal the official test class
  counts and whose percentages round to the published recall/precision values.
* Exact bit-reproducibility is only guaranteed for the same scikit-learn/libsvm version.

---

## Citation

If you use this recreation, please cite the original study:

> D. Anguita, A. Ghio, L. Oneto, X. Parra and J. L. Reyes-Ortiz. *A Public Domain Dataset
> for Human Activity Recognition Using Smartphones.* 21st European Symposium on Artificial
> Neural Networks, Computational Intelligence and Machine Learning (ESANN 2013), Bruges,
> Belgium, 24-26 April 2013, pp. 437-442.

> Reyes-Ortiz, J., Anguita, D., Ghio, A., Oneto, L., & Parra, X. (2013).
> *Human Activity Recognition Using Smartphones* [Dataset]. UCI Machine Learning
> Repository. <https://doi.org/10.24432/C54S4K>

The dataset is distributed AS-IS; any commercial use is prohibited by its original
license terms, while the UCI repository entry lists it under CC BY 4.0 with attribution.


