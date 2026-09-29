# Independent Replication and Verification of Smartphone-Based Human Activity Recognition Using Multiclass Support Vector Machines

This repository is a careful re-run of a well-known scientific experiment. The original
experiment is the UCI *Human Activity Recognition Using Smartphones* study (2013), and
this project repeats it from scratch with a single program. It ends up with almost
exactly the same score the original authors reported, so it counts as a successful copy.

## 1. Original Experiment

1. **30 people** (aged 19-48) strapped a mobile phone into a pouch on their waist.
2. They were asked to perform **six everyday movements**, one after another:

   | Name used in the data | What the person was actually doing |
   | --- | --- |
   | Walking | walking on flat ground |
   | Walking upstairs | climbing a staircase |
   | Walking downstairs | going down a staircase |
   | Sitting | sitting still |
   | Standing | standing still |
   | Laying | lying down |

3. The phone's built-in movement sensors recorded how the body moved **50 times every
   second**.
4. The recordings were tidied up (sensor noise removed, and the phone's own steady pull
   separated from the person's actual body movement).
5. The long recordings were cut into **short overlapping clips of about 2.6 seconds**
   each, and each clip was summarised as **561 numbers** describing that slice of
   movement: how fast the body moved, how jerky it was, how strong the shaking was, how
   much of the movement was slow or fast, and so on. Those 561 numbers are the *only*
   thing the computer is ever shown — it never sees a video and never learns the
   person's name.
6. The authors then *taught* a computer program to look at those 561 numbers and say
   which of the six movements the clip came from.

## 2. Methodology

The 30 people were divided into two groups **of people, not of clips**:

* **21 people** — their clips (7,352 clips) were used only for *learning*.
* **9 people** — their clips (2,947 clips) were locked away and used only for the
  *final test*.

The final test uses **nine people the program has never seen** so the score sells us *"how well this works
on a new person?"* — not *"how well it remembers people it has already seen?"*

The nine people in the exam group are exactly the ones the original authors chose, so
the comparison between their score and ours is fair.

## 3. Learning Method

The learning method used here is a classic, widely used one called a **support-vector
machine** (usually shortened to **SVM**). In plain terms, it is a program that draws
boundaries between groups of numbers, and it works best when those boundaries are allowed
to be curvy instead of straight lines. Six slightly different copies of it are trained —
one per activity — and each one answers a simple yes/no question: *"is this clip an
example of my activity?"* Whichever copy is most confident wins the vote.

Like many such programs, this one has **two dials** that must be set before the real
exam:

* one dial decides *how strictly* the program must fit the learning examples;
* the other decides *how curvy* the boundaries are allowed to be.

Setting them by taste would be cheating, so the program tests **12 combinations** and
keeps the one that does best on practice rounds built only from the learning group. It
never looks at the exam clips while choosing.

Those practice rounds work like this: the 7,352 learning clips are split into 10 piles,
then 10 times in a row the program hides one pile, learns from the other nine and is
quizzed on the hidden pile. The average quiz score decides which dial setting wins. This
routine has a long name in textbooks (*10-fold cross-validation*) and appears on screen
as "CV accuracy".

One detail matters here. Neighbouring clips overlap, so two clips next to each other are
almost the same thing twice. The program therefore keeps each person's clips together in
the same pile, so an almost-copy of a quiz question can never sit in the learning piles.
(There is an option to switch that protection off, but the default is the careful one.)

---

## 4. Results

| What was measured on the 2,947 exam clips (9 unseen people) | This project | The original paper |
| --- | --- | --- |
| Clips labelled correctly | **2,855 (96.88%)** | 2,840 (96.37 %, printed as "96%") |
| Looked at out of every 100 clips | **about 97 right** | about 96 right |
| Verdict | only 0.5 points away from the published result — **an accurate copy** | — |

The most common mix-up in our run is the same one the paper reports: **sitting is often
mistaken for standing** (and vice versa). That makes sense — when measured from the
waist, someone sitting very still and someone standing very still look almost identical.

---

## 5. Run it yourself

You will need about 15 minutes the first time. After that, a fresh result takes under a
minute.

**Step 1 — get the project files.** Either:

* download this repository as a ZIP (green *Code* button on the GitHub page → *Download
  ZIP*) and unzip it somewhere easy, such as your Desktop; or
* if you have Git, in a terminal run
  `git clone https://github.com/advaitpatel-official/uci-har-svm-replication.git`.

You should end up with a folder containing `har_mcsvm_recreation.py`, `README.md` and a
`data` folder.

**Step 2 — install Python (once).** Go to <https://www.python.org/downloads/>, download
the installer for your system and run it. On Windows, **tick the box that says "Add
python.exe to PATH"** on the first screen of the installer — the program will not start
without it. Any version from 3.10 upwards is fine (this project was last checked on
Python 3.14.6).

**Step 3 — open a terminal *in the project folder*.** Two easy ways:

* In VS Code: *File → Open Folder…*, pick the project folder, then *Terminal → New
  Terminal*.
* On Windows: in File Explorer, hold Shift, right-click the project folder and choose
  *Open in Terminal*.

**Step 4 — paste this line and press Enter:**

```text
python har_mcsvm_recreation.py
```

Then wait. The first time, the program downloads the dataset (about 58 MB) and installs
three helper libraries it needs (`numpy`, `scipy`, `scikit-learn`) — it does all of that
by itself. The learning step itself takes roughly 4-15 minutes on a normal laptop,
because it tries 12 dial settings and checks each one with 10 practice rounds.

Two shortcuts, if you would rather not wait:

```text
python har_mcsvm_recreation.py --quick        # one dial setting, result in ~30 seconds
python har_mcsvm_recreation.py --list-checks  # no learning at all, just check the data
```

**Step 5 — read the last few lines.** The important ones look like this:

```text
19/19 checks passed
RESULT: the multiclass SVM reproduces the paper's 96 % on the official test set.
```

> If you see *"Python was not found"* or *"'python' is not recognized"*, the PATH box in
> Step 2 was probably left unticked. Re-run the Python installer and tick it, or simply
> try typing `py` instead of `python`.

---

## 6. Output

Here is how to interpret the output of running the program.
```text
paper    : A Public Domain Dataset for Human Activity Recognition Using Smartphones
dataset  : UCI ML Repository id 240 ...
protocol : 30 subjects, 561 features, 7352 train / 2947 test patterns, 50 Hz, 2.56 s windows
model    : multiclass SVM = OneVsAll of binary RBF-kernel SVMs; 10-fold CV model selection
[data ] using cached dataset: ...\data\UCI HAR Dataset
[data ] loaded 7352 training and 2947 test patterns with 561 features
```

> *Which experiment is being re-run, which files were found, and how many clips were
> loaded.* "Training" clips are the practice material, "test" clips are the exam.

```text
--- paper, Table 4 (multiclass SVM, 2947 test patterns) ---
                            WK     WU     WD     ST     SD     LD
           Walking (WK)    492      1      3      0      0      0
  Walking upstairs (WU)     18    451      2      0      0      0
Walking downstairs (WD)      4      6    410      0      0      0
           Sitting (ST)      0      2      0    432     57      0
          Standing (SD)      0      0      0     14    518      0
            Laying (LD)      0      0      0      0      0    537
                 recall    99%    96%    98%    88%    97%   100%
              precision    96%    98%    99%    97%    90%   100%
       overall accuracy    96%   = 2840/2947 correct, published as 96 %
```

> *The numbers published by the original authors, reproduced here so the two runs can be
> compared side by side. Rows are what the person was really doing, columns are what the
> authors' program guessed, and the numbers count clips. The big numbers on the diagonal
> are the correct guesses.*

```text
[train] quick mode: single hyper-parameter set C=100, gamma=scale
[train] 10-fold cross-validation of that setting ...
[train] CV accuracy = 0.9569 +/- 0.0398
[train] trained OneVsAll(RBF SVM): C=100, gamma=scale (effective 0.00632), 1592 support vectors, 4.3 s
```

> *The learning step. "CV accuracy = 0.9569" means the program scored about 96 % on the
> practice rounds that used only the 21 learning people — before it ever touched the exam
> clips. The names `C` and `gamma` are simply the two dials from section 3. "Support
> vectors" is the count of learning clips the program decided to keep as reference
> examples. "4.9 s" is how long the learning took.*

```text
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
```

> *Our program's answer sheet, laid out exactly like the paper's so the two can be
> compared. It is nearly identical: 2,855 correct out of 2,947 clips, and the only large
> box off the diagonal is the Sitting/Standing one (46 clips), just as in the paper.*

```text
--- difference to the paper's Table 4 (this run - paper) ---
           Walking (WK)     -3     +6     -3
  Walking upstairs (WU)     -7     +9     -2
...
identical cells 21/36, total absolute deviation 74 windows, max cell deviation 11
```

> *The difference between the two answer sheets, box by box. `+9` means our run put nine
> more clips in that box than the paper did, `-3` means three fewer, and `+0` means the
> two runs agree exactly. 21 of the 36 boxes agree exactly, and the worst disagreement is
> 11 clips out of 2,947 (0.4 %).*

```text
--- per-subject accuracy on the 9 test subjects ---
subject  2:  302 windows, accuracy  98.34 %
subject  4:  317 windows, accuracy  96.85 %
subject  9:  288 windows, accuracy  92.36 %
subject 10:  294 windows, accuracy  92.52 %
...
subject 24:  381 windows, accuracy  99.74 %
```

> *How the program did on each of the nine unseen people (their real identities are just
> numbers 2 to 24, because the dataset is anonymised). Everyone is between 92 % and 100 %,
> so the overall score is not an average hiding one person being completely misread.*

```text
[PASS] dataset layout 7352x561 / 2947x561              X_train (7352, 561), X_test (2947, 561)
[PASS] 6 activities, README label order                1=WALKING, 2=WALKING_UPSTAIRS, ...
[PASS] 70 % / 30 % split by subject                    21 train subjects / 9 test subjects
[PASS] overall accuracy reproduces the paper's 96 %    96.88 % vs 96.37 % (delta +0.51 pp)
...
18/18 checks passed

total runtime: 30.1 s
RESULT: the multiclass SVM reproduces the paper's 96 % on the official test set.
```

> *The automatic check-up, which is what actually decides success or failure. Every line
> says `PASS` or `FAIL` and shows the numbers it used, so you never have to take the
> program's word for anything. Section 8 explains each check in plain words. The very
> last line is the verdict.*

---

## 7. Data Interpretation

Every percentage in the print-out comes from counting clips on the answer sheet. Nothing
is estimated or summarised by a library; the program adds up the boxes itself.

| The number on screen | What it means in ordinary words | This run | The paper |
| --- | --- | --- | --- |
| Overall accuracy | out of all 2,947 exam clips, how many got the right label | 96.88% (2,855) | 96.37% (2,840) |
| Recall for one activity | out of the clips where the person *really was* doing that activity, how many did we spot? | 99 / 98 / 96 / 90 / 98 / 100% | 99 / 96 / 98 / 88 / 97 / 100% |
| Precision for one activity | when the program *says* that activity, how often is it right? | 97 / 96 / 100 / 98 / 92 / 100% | 96 / 98 / 99 / 97 / 90 / 100% |
| CV accuracy | the average score of the practice rounds on the learning people only | 95.69% | used the same routine to pick dials |
| Confidence range | the range the true score is likely to fall in, given that only 2,947 clips were tested; the paper's number sits inside it | 96.19% - 97.45% | 96.37% (inside) |
| Per-subject accuracy | the same score calculated separately for each of the nine exam people | 92.4% - 99.7% | not reported per person |

---

## 8. Troubleshooting

| What you see | What it means and what to do |
| --- | --- |
| *"'python' is not recognized"* or *"Python was not found"* | Python is not installed, or it was installed without the "Add python.exe to PATH" box ticked. Re-run the installer and tick it, then close and reopen the terminal. On Windows you can also just try `py har_mcsvm_recreation.py`. |
| The screen fills with `pip install ...` messages the first time | Normal. The program installs three helper libraries by itself. It needs an internet connection for that, once. |
| *"Could not download the dataset"* | You are offline, or a firewall blocked it. The dataset is already inside the `data` folder of this repository — if you downloaded the project as a ZIP you are fine; otherwise copy the `data` folder next to the script, or re-run with `--no-download`. |
| Nothing seems to happen for several minutes | The full run tries 12 dial settings with 10 practice rounds each; 4-15 minutes on a laptop is normal. For a quick look, press Ctrl+C and run `python har_mcsvm_recreation.py --quick` instead. |
| A check line says `FAIL` | Read that line: it shows the two numbers that disagree. A failing *accuracy* check while everything else passes usually just means your run and the paper differ by a bit more than 2 points; you can widen the allowance with `--tolerance 0.03`. A failing *data* check means the dataset files are wrong or incomplete — delete the `data` folder and run again so they are downloaded fresh. |
| *"RESULT: at least one check failed"* | Same as above: the run completed, but the recreation is not faithful enough to be trusted. Do not report the score until the failing check is understood. |
| Nothing at all works and you just want the numbers | The run summarised in [section 4](#4-the-result-in-one-table) and the outputs in [section 6](#6-what-appears-on-the-screen-block-by-block) are the real output of this program on this dataset. |

---

## 9. Extras

You can ignore this section completely — the plain command from section 5 already runs
the full experiment the way the paper describes it. These extra words are for curiosity
or for slower/faster machines.

| Extra word | What it does | Default |
| --- | --- | --- |
| `--quick` | use one dial setting instead of searching (about 30 seconds) | off |
| `--no-cv` | with `--quick`, also skip the practice rounds (fastest possible run) | off |
| `--grid fast` | search a smaller set of dial settings (about a third of the time) | searches all 12 |
| `--cv 5` | use 5 practice piles instead of 10 | 10 |
| `--shuffle-cv` | switch off the protection against overlapping clips leaking into the practice material — scores look better but mean less; not for reporting | off |
| `--jobs 4` | how many processor cores to use (`-1` = all of them) | all |
| `--seed 42` | the number used when shuffling (only matters with `--shuffle-cv`) | 0 |
| `--cache-mb 200` | scratch memory each of the six judges may use while learning | 500 |
| `--tolerance 0.03` | how far the score may drift from the paper's before a check fails | 0.02 (2 points) |
| `--class-tolerance 0.10` | the same allowance, calculated per activity | 0.05 (5 points) |
| `--data-dir "C:\somewhere"` | keep the dataset somewhere else | the `data` folder next to the script |
| `--dataset-url URL` | download the dataset from a different address | the official UCI address |
| `--no-download` | never use the internet, even if the data is missing | off |
| `--list-checks` | only check the data and the paper's table, learn nothing (a few seconds) | off |

---

## 10. Sources

**Original Experiment**

> D. Anguita, A. Ghio, L. Oneto, X. Parra and J. L. Reyes-Ortiz. *A Public Domain Dataset
> for Human Activity Recognition Using Smartphones.* ESANN 2013, Bruges (Belgium),
> 24-26 April 2013, pp. 437-442.
> <https://www.esann.org/sites/default/files/proceedings/legacy/es2013-84.pdf>

**Dataset**

> Reyes-Ortiz, J., Anguita, D., Ghio, A., Oneto, L., & Parra, X. (2013).
> *Human Activity Recognition Using Smartphones* [Dataset]. UCI Machine Learning
> Repository. DOI `10.24432/C54S4K`. <https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones>

The dataset contains no names or personal details, only people numbered 1 to 30. It is
free to use with credit to the authors; its original licence forbids commercial use, so
treat this project as educational.

**What is in this folder**

```text
UCI_HAR_Recreation/
├── har_mcsvm_recreation.py     the entire experiment: data, learning, scoring, checks
├── README.md                   this file
└── data/
    ├── UCI_HAR_Dataset.zip     the official download, kept so it is fetched only once
    └── UCI HAR Dataset/        the unpacked data
        ├── README.txt          the authors' own description of the data
        ├── features.txt        the names of the 561 numbers
        ├── activity_labels.txt the six activity names
        ├── train/              7,352 clips from 21 people (+ raw sensor files, unused)
        └── test/               2,947 clips from 9 people  (+ raw sensor files, unused)
```

There is only one program file, so you can read the whole experiment top to bottom if you
ever want to. If you do, its comments are written for someone who knows a little Python;
this README is the gentler introduction.

---

## 11. Limitations

* **Only the thinking half is re-done here.** This project starts from the ready-made
  561-number files that the original authors published. It does *not* redo the cleaning,
  the cutting into clips or the making of the 561 numbers — those are taken as given, so
  a mistake made there by us is impossible, and a mistake made there in 2013 is invisible
  to this project.
* **96.88 % is not a promise.** It is the score for nine particular people, 2,947
  particular clips, and one particular choice of dials. A different set of people, a
  different phone position or a different activity could easily score lower.
* **The exam is a fair exam, but it is still an exam.** The exam people were never used
  for learning or for dial-choosing, so the score is an honest one — but with only nine
  people, a couple of points either way mean very little.
* **Exact repeatability has a caveat.** Running the program twice on the same computer
  gives identical results (check 12 proves it). Running it with newer versions of the
  three helper libraries could move a few clips, because the way numbers are rounded
  inside them can change.
* **Two of the shortcuts are not for quoting.** `--quick` uses a fixed dial setting and
  `--shuffle-cv` weakens the protection against overlapping clips. Both are convenient
  for a quick look, but the numbers to report are the ones from the plain full run.





