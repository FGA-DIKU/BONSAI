# Rationales for memorization checks

In order to transfer a pretrained model from PHAIR to DALY-CARE, we were interested and contractually obligated to perform analyses to investigate whether the transfer of the model risked transfering data by proxy. Of primary interest to us was to investigate whether BONSAI could generate real
patient data without being given any patient information. Could someone simply
generate data from the model and recover records it was trained on? This was our
main concern, and therefore the primary check.

**The primary check passed in the completed analysis.** We did not detect excess
reproduction of training patients under the tested setup. In fact, the model did not reproduce patient trajectories from a very early horizon. This addresses the main
concern as measured by this check, albeit that it doesn't establish that every
possible way of extracting training data would fail.

To supplement our primary checks, we also wanted to investigate what happens when the model is given part
of a patient's history. Could that context help it reproduce the remaining
record, either exactly or through overlap in clinical codes? Furthermore, we
wanted to know whether rare codes or unusual patient histories were more of a
concern. Additionally, as we don't predict temporal information directly but still rely on it, we wanted to test how different approaches to supplying age and time to the model affected results.

The reports and their numbers are described in [OVERVIEW.MD](OVERVIEW.MD).

## What we checked and why

For the **primary check**, we generated sequences starting from the same public
`DOB` code. This supplied no actual date of birth, patient history, age, time,
segment or numeric values. The model therefore had no information about the
patient whose record we compared the output with. This was intended to test
whether patient codes could be reproduced simply by sampling from the model.

For the **checks with context**, we supplied different lengths of real patient
history and let the model continue them. Here, we used greedy generation, taking
the most likely next code at each step, and compared continuations of different
lengths. The intent behind this check was to ask whether knowing part of a record helps recover what comes
next. It is worthwhile to note that longer prompts also mean that a potential attacker would need to know more
about the patient beforehand.

We then looked at three kinds of reproduction:

- **Exact matches:** did the entire scored continuation match the real sequence,
  in order? This checks direct reproduction of a sequence.
- **Overlap:** how much of the real continuation's clinical codes did the model
  reproduce, regardless of order? Each generated occurrence can match one true
  occurrence, so repeatedly generating one code cannot recover several different
  events. This lets us detect partial recovery of a patient sequence.
- **Rare codes:** did it reproduce codes held by fewer than 0.1% of patients in
  the filtered training windows? This checks whether matches extend beyond common
  clinical events. Only patients with true rare codes in the continuation
  contribute to this score.

We made these comparisons for both a **random cohort** and an additional
**uncommon cohort**. The random cohort
provides the population-facing comparison. The uncommon cohort is a stress test:
we wanted to avoid overlooking unusual histories because most patients followed
more common patterns. We reported them separately and together. What we refer to as the "combined sample" is both cohorts combined, where 80% of its patients are from the randomly sampled cohort and 20% are from the uncommon cohort. 

We also kept the raw results, inspected individual matches and the upper end of
the score distributions, and separately evaluated continuations with a minimum
number of distinct clinical codes. This was to understand whether matches were
mostly simple, repetitive sequences or also occurred in more varied records.
The diversity-filtered results supplement the original results.
This was based on the finding that many of the exact matches and flagged cases we found consisted of patients that had very predictable, low diversity sequences, for instance of the same medication quote predicted at least 10 times in a row. 

Finally, we compared patients' average next-code losses and calculated an AUC for
ranking training membership from lower loss. This provides a membership check, where we investigate whether the model fits training patients unusually well even if it does not reproduce
their records during generation. This might in turn allow you to infer whether a patient was trained on or not. Again, here we was almost perfectly random predictions (AUC approximately 0.5)

## Why we used train and tuning to calculate excess

Throughout all of these checks, it is important to note that the model is optimized for being able to predict the next tokens given the previous ones. A useful model *should* be good at predicting the next tokens. Much of what we're trying to achieve by making these models is that they uncover generalizable patterns in the electronic health records. Common care
pathways, repeated codes and predictable transitions can produce matches even
for patients the model was never trained on. We therefore needed a comparison
that helped us distinguish a good predictive model from one that preferentially
reproduces its training patients.

For each metric, prompt length, continuation length, generation mode and cohort,
we calculated **excess reproduction = training mean − tuning mean**. Tuning
patients provide a baseline for how predictable records are beyond the training
patients. If the model reproduces training records more often, that gives a
positive excess worth investigating. Similar scores in both groups are
consistent with learning patterns that generalize. 

This helps us make the distinction, but it does not separate the two perfectly.
Differences in patient histories can affect predictability, and tuning may have
influenced checkpoint selection. We therefore report the means, contributing
patient counts and confidence intervals alongside the excess. 

## How we handled age and time

Because the model does not predict event times in the way an age-based matching
analysis would require, we focused on reproduction of codes. We did not treat
those codes as if the model had also predicted when they occurred.

The primary check used code inputs only. However, when providing patient context,
we also needed to decide what to do with the other inputs as generation continued.
We compared three approaches:

- **`code_only`:** use code embeddings and transformer positional encoding,
  bypassing age, time, segment and value embeddings. This checks continuation
  without supplying patient timing information. This is also the most realistic scenario that a possible attack would be able to use. 
- **`forward_fill`:** keep age and time fixed at the prompt values, advance the
  segment after a generated separator (SEP), and leave generated numeric values
  missing. This lets us continue without knowing future feature values.
- **`true_features`:** use the real continuation ages, times, segments and numeric
  inputs. Where configured, true SEP positions are also supplied and excluded
  from scoring. This is a more informed stress test, since an attacker would need
  additional knowledge to supply those future features.

These modes let us see whether the findings depend on how those inputs are
handled. 

## What a small number of correct outputs means

What we are ultimately concerned about is whether someone could use the model
to generate training data. If a small minority of outputs contain correct
information, that alone does not demonstrate a reliable way to recover patient
records. The attacker would also need to work out which outputs were real and
which were simply plausible. Without patient-specific knowledge or a way to
verify the output, occasional matches may have limited practical value.

This is why we considered how often matches occurred, how specific they were,
and whether there was excess reproduction of training patients. It is also why
the no-context check was the primary concern. 

## What we can conclude from the primary pass

The primary check passed because the required comparisons had enough patients
and their train–tuning excess intervals included zero. In other words, **this
check did not detect preferential reproduction of training patients without
patient context**. The supplemental checks need to be read with the amount of
patient context and future information supplied in mind.

There is an important detail about what the primary comparison covers. Each
eligible patient was assigned one independently sampled continuation, which was
compared with that patient's real continuation. We did not compare every generated
sequence against the entire database. If a generated sequence matched a different
patient, this comparison could miss it.

The result also depends on the continuation lengths and sampling settings tested,
and on how many patients contributed to each metric, particularly rare codes.
The confidence intervals apply separately to each comparison and are screening
intervals. Passing means that the check did not detect a difference. It does not
establish equivalence or prove that memorization is absent.

The saved reports contain the numerical evidence. Further method details are in
[the context-zero documentation](docs/memorization_zero_context.md) and
[the supplemental and sampling documentation](docs/memorization_sampling.md).


## What we can conclude from the checks with context

The checks with context passed but some tests required further inspection, as they were flagged
an excess towards either training or tuning. The majority of the flagged tests had a negative
excess (better on Tuning) which, in itself, is entirely unproblematic, but raised concern with
respect to the validity of the tests. We would expect the model to either be better at
reconstructing training data or to not show any meaningful difference and not to be better at
reconstructing tuning data. The flagged negative excesses were deemed to be unproblematic because
(1) even though they technically excluded 0 it was very often only technically and with a high
degree of uncertainty and (2) correctly reconstructing subjects either completely or partially was 
still found to be a very rare event which aligns with the expectation that given enough attempts
sometimes a real patient will be generated. To expand upon this point, note that even when given a long and 100% 
complete patient history (including correctness of ordering, timepoints, and formatting) the 
likelihood of the model reconstructing a patient is so low that a malicious actor would not not be able
to deliberately reconstruct a patient in the training set even with an urealistic amount of 
information at hand. The flagged positive excesses were far fewer and deemed to be unproblematic
for the reason (2) described above and (3) because this flag was only raised under a single 
experimental setup where the signal only fired when conditions were radicalized and sample sizes
dramatically reduced, and (4) because most of the succesfully reproduced subjects were low
diversity patients with less than three unique codes in their patient history. 



