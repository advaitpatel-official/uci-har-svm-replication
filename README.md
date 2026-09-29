# Can a phone tell what you are doing?

**Short answer: yes — and this project shows it. A computer can be taught to recognise
walking, climbing stairs, sitting, standing and lying down from the movement of a phone
worn on the waist, and it gets it right about 97 times out of 100.**

This repository is a careful re-run of a well-known scientific experiment. The original
experiment is the UCI *Human Activity Recognition Using Smartphones* study (2013), and
this project repeats it from scratch with a single program. It ends up with almost
exactly the same score the original authors reported, so it counts as a successful copy.

**You do not need to know how to program to use this.** You only need to copy and paste
two lines into a black window on your computer (called a *terminal*) — there is a
step-by-step guide in [section 5](#5-run-it-yourself-step-by-step). Everything else in
this file explains what the program is doing and what the numbers on the screen mean.

If you are looking for the mathematics and the formal experiment details, they belong to
the [original paper](#13-where-the-paper-and-the-data-come-from), not to this README —
this file is written for a person who has never written a line of code.

**What is in this file**

1. [What the original experiment did](#1-what-the-original-experiment-did)
2. [Why the result is believable](#2-why-the-result-is-believable-the-clever-part-of-the-design)
3. [How the program learns](#3-how-the-program-learns-and-how-its-settings-get-chosen)
4. [The result, in one table](#4-the-result-in-one-table)
5. [Run it yourself (step by step)](#5-run-it-yourself-step-by-step)
6. [What appears on the screen, block by block](#6-what-appears-on-the-screen-block-by-block)
7. [What the numbers mean, in ordinary words](#7-what-the-numbers-mean-in-ordinary-words)
8. [How we know the numbers are not faked](#8-how-we-know-the-numbers-are-not-faked)
9. [Why our numbers differ slightly from the paper's](#9-where-our-numbers-differ-from-the-papers-and-why-that-is-normal)
10. [If something goes wrong](#10-if-something-goes-wrong)
11. [Every technical word explained](#11-words-that-appear-on-screen-in-plain-language)
12. [Optional extra settings](#12-optional-the-buttons-you-can-press)
13. [Where the paper and the data come from](#13-where-the-paper-and-the-data-come-from)
14. [Honest limits of this recreation](#14-honest-limits-of-this-recreation)

---

## 1. What the original experiment did

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

In everyday words: one clip plus its correct answer is a **worked example**. All the
clips together are called the **dataset**. Teaching a program from worked examples like
this is what people usually mean by "artificial intelligence" or "machine learning".

## 2. Why the result is believable (the clever part of the design)

The 30 people were divided into two groups **of people, not of clips**:

* **21 people** — their clips (7,352 clips) were used only for *learning*.
* **9 people** — their clips (2,947 clips) were locked away and used only for the
  *final test*.

That is the difference between practising on questions you have already seen and sitting
a real exam on questions you have not. Because the final test uses **nine people the
program has never met**, the score answers an honest question: *"how well does this work
on a new person?"* — not *"how well does it remember people it has already seen?"*

The nine people in the exam group are exactly the ones the original authors chose, so
the comparison between their score and ours is fair.

## 3. How the program learns, and how its settings get chosen

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

## 4. The result, in one table

| What was measured on the 2,947 exam clips (9 unseen people) | This project | The original paper |
| --- | --- | --- |
| Clips labelled correctly | **2,855 (96.88 %)** | 2,840 (96.37 %, printed as "96 %") |
| Looked at out of every 100 clips | **about 97 right** | about 96 right |
| Verdict | only 0.5 points away from the published result — **an accurate copy** | — |

The most common mix-up in our run is the same one the paper reports: **sitting is often
mistaken for standing** (and vice versa). That makes sense — when measured from the
waist, someone sitting very still and someone standing very still look almost identical.

---

## 5. Run it yourself (step by step)

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

That is the whole "did it work?" answer: every check passed and the score matches the
published one. If you instead see `RESULT: at least one check failed`, the line just
above it names the check that failed and shows the numbers involved — see
[section 10](#10-if-something-goes-wrong).

> If you see *"Python was not found"* or *"'python' is not recognized"*, the PATH box in
> Step 2 was probably left unticked. Re-run the Python installer and tick it, or simply
> try typing `py` instead of `python`.

---

## 6. What appears on the screen, block by block

Below is one real, short run (the `--quick` version, trimmed to the interesting parts)
with a plain-English note after each block.

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

## 7. What the numbers mean, in ordinary words

Every percentage in the print-out comes from counting clips on the answer sheet. Nothing
is estimated or summarised by a library; the program adds up the boxes itself.

| The number on screen | What it means in ordinary words | This run | The paper |
| --- | --- | --- | --- |
| overall accuracy | out of all 2,947 exam clips, how many got the right label | 96.88 % (2,855) | 96.37 % (2,840) |
| recall for one activity | out of the clips where the person *really was* doing that activity, how many did we spot? | 99 / 98 / 96 / 90 / 98 / 100 % | 99 / 96 / 98 / 88 / 97 / 100 % |
| precision for one activity | when the program *says* that activity, how often is it right? | 97 / 96 / 100 / 98 / 92 / 100 % | 96 / 98 / 99 / 97 / 90 / 100 % |
| CV accuracy | the average score of the practice rounds on the learning people only | 95.69 % | used the same routine to pick dials |
| confidence range | the range the true score is likely to fall in, given that only 2,947 clips were tested; the paper's number sits inside it | 96.19 % - 97.45 % | 96.37 % (inside) |
| per-subject accuracy | the same score calculated separately for each of the nine exam people | 92.4 % - 99.7 % | not reported per person |

A quick way to read the two tables of numbers:

* **Percentage on the diagonal is boring and good** — those are correct guesses.
* **Everything off the diagonal is a mix-up.** The program said column, the truth was
  row.
* Row 4, column 5 in both tables (46 in ours, 57 in theirs) is the famous
  Sitting-vs-Standing confusion: two still poses that look almost the same when you only
  measure the waist.

---

## 8. How we know the numbers are not faked

A score on its own proves nothing — anyone can print "96 %". So the program ships with
**19 automatic checks** that either pass or fail, and the run reports how many of them
passed. The count depends on how much work it did:

| Situation | Checks run |
| --- | --- |
| `--list-checks` (no learning at all) | 11 of 19 |
| `--quick` (one dial setting) | 18 of 19 |
| full run (dial search) | 19 of 19 |

The remaining checks only exist when the program has actually learned something — for
example "did the score match the paper's?" cannot be answered before there is a score.

**The 11 checks that only look at the data**

1. the two data files have exactly the expected shapes (7,352 training clips and 2,947
   test clips, each described by 561 numbers);
2. all 561 columns are present and named;
3. the six activity names are the right six, in the right order;
4. no broken or missing numbers anywhere, and all values inside the allowed range;
5. the number of clips for each activity matches the published counts;
6. the 30 people were really split 21 / 9, with nobody appearing in both groups;
7. the nine exam people are exactly the ones the paper used (people 2, 4, 9, 10, 12, 13,
   18, 20 and 24);
8. the clip length and overlap match what the paper describes;
9. the paper's printed table adds up to the published clip counts (so we read it
   correctly);
10. the paper's table really does produce the published per-activity percentages; and
11. the paper's table really does produce the published 96 % — in other words, the target
    we are aiming at is the target the paper published (2,840 correct out of 2,947).

**The 7 extra checks added once the program has learned**

12. **the result is repeatable:** learning the same thing a second time gives exactly the
    same answers, clip for clip — so the score is not luck;
13. **we hit the target:** our score is within 2 points of the paper's (adjustable);
14. **the gap is not meaningful:** the paper's number lies inside the range of scores our
    run could plausibly have produced with only 2,947 test clips;
15. **we beat the older method** the paper compares itself against (90.8 %);
16. **we match activity by activity**, within 5 points of the paper's per-activity
    percentages (adjustable);
17. **we make the same mistake as the paper:** Sitting/Standing is the worst mix-up here
    too;
18. **the practice score and the exam score agree** (95.7 % vs 96.9 %) — a big gap would
    suggest the dials were tuned to the exam, which would make the score meaningless.

**The final check, only in a full run**

19. **the dials chosen really are the best of the 12** that were tried — the same rule the
    paper used. (`--quick` skips the search, so there is nothing to check.)

Why bother with all this? Because picture-perfect reproduction is not the goal — *honest*
reproduction is. The checks make it hard for a mistake or an accidental shortcut to hide.
If a check fails, the program says so in plain words and exits with an error signal
instead of quietly printing a nice-looking percentage.

| Exit signal | Meaning in plain words |
| --- | --- |
| `0` | everything passed — the recreation worked |
| `1` | at least one check failed (details are on screen) |
| `2` | the data could not be found or downloaded, so nothing ran |

---

## 9. Where our numbers differ from the paper's, and why that is normal

Our run and the paper's run are two different programs. Same idea, same data, same rules
— but not the same lines of code. The original authors wrote their own implementation of
the method (their later work even re-writes it for low-power phone chips), whereas this
project uses a standard, widely used open-source library. Tiny differences in how numbers
are rounded inside such a program can move a handful of clips from one box to another.

What that looks like in practice: 15 of the 36 boxes in the answer sheet differ slightly,
most by one or two clips, with a total of 74 clips out of 2,947 (2.5 %) placed differently
and 0.5 points of overall score. Crucially, the *shape* of the mistakes is the same — the
same activities are easy, the same pair is hard, and the same nine people are the exam.

One more honest caveat: the paper is a PDF, and the table was read out of its text. Where
the layout was ambiguous, the program keeps the reading that is consistent with the
published totals and percentages — that is what checks 9 to 11 above test.

---

## 10. If something goes wrong

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

## 11. Words that appear on screen, in plain language

| Word | What it actually means |
| --- | --- |
| dataset | the collection of recordings and their correct answers |
| clip, window, example, pattern | one ~2.6 second slice of movement and the activity it belongs to |
| feature | one of the 561 numbers that describe a clip |
| model, classifier, trained program | the thing after learning — the program that now guesses activities |
| learning, training, fitting | adjusting the program using the 21 people's clips |
| test set, exam clips | the 9 people's 2,947 clips, locked away until the very end |
| practice rounds, cross-validation ("CV") | splitting the learning clips into 10 piles so the program can quiz itself honestly |
| dials, hyper-parameters, `C` and `gamma` | the two settings that must be chosen before learning; `C` = how strictly to fit the examples, `gamma` = how curvy the boundaries may be |
| support vector | a learning clip the program kept as a reference example |
| accuracy | share of all exam clips labelled correctly |
| recall | share of the clips of one activity that were actually spotted |
| precision | share of the clips labelled as one activity that really were that activity |
| confusion matrix, answer sheet, Table 4 | the two-way tally of "true activity" versus "guessed activity" |
| confidence interval | the range the true score plausibly lies in, given the limited number of test clips |
| library, package | ready-made code written by other people that this program re-uses (here: `numpy`, `scipy`, `scikit-learn`) |
| terminal, shell, PowerShell, command line | the black text window where you type commands |
| flag, option, `--quick` and friends | extra words added after the command name to change how it runs |
| exit code | the tiny number the program hands back to the computer to say "all good" (0) or "something failed" (1 or 2) |

---

## 12. Optional: the buttons you can press

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

## 13. Where the paper and the data come from

**The experiment being re-run**

> D. Anguita, A. Ghio, L. Oneto, X. Parra and J. L. Reyes-Ortiz. *A Public Domain Dataset
> for Human Activity Recognition Using Smartphones.* ESANN 2013, Bruges (Belgium),
> 24-26 April 2013, pp. 437-442.
> <https://www.esann.org/sites/default/files/proceedings/legacy/es2013-84.pdf>

**The data**

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

## 14. Honest limits of this recreation

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





