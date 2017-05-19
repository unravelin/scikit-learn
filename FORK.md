# What is this?

This is a fork of sklearn0.19dev (based off 676f878e9815dee360e8ea6abb5d233cc82af025).

It incorporates raghavrv's work outlined [here](https://github.com/scikit-learn/scikit-learn/pull/5974)
and the latest rebase of his work will live in `rf_missing` branch of this repository.

# Why the fork?

Dealing with missing values on this release is explicit. If you instantiate
`EnsembleInspector` with `missing_values=-99` then `-99` will be treated as
missing sentinel when training. Resulting trees will have `missing_value`
np.array attribute that will contain missing direction for each node (left/right).
Missing direction is the direction taken during the prediction if the value is
missing.

If you instantiate `EnsembleInspector` with `missing_values=None` then explicit
dealing with missing values will be turned off. Each tree's `missing_value`
attribute will be empty. This is equivalent to training on unmodified sklearn.

Whilst the improvement of accuracy gained from dealing with missing values
explicitly is marginal, it gives you greater control over how to handle missing
during predictions.

# How do I install this?

If you are on pip version 9 or later, then running `pip install -r requirements.txt`
where `requirements.txt` starts with

```
--trusted-host piprepo.ravdns2.co.uk
--extra-index-url https://piprepo.ravdns2.co.uk
scikit-learn==0.19.1+ravelin
```

should work. However due to an upstream [issue in pip](https://github.com/pypa/pip/issues/3799) you must run
`pip3 install --trusted-host piprepo.ravdns2.co.uk -r requirements.txt` until it's fixed

If you want to avoid using `requirements.txt` file then run the following:

`pip3 install --trusted-host piprepo.ravdns2.co.uk --extra-index-url https://piprepo.ravdns2.co.uk scikit-learn==0.19.1+ravelin`

# How to make a release?

WIP

# How it works?

WIP

Explained by the author:

```
A brief explanation of my approach.

For each feature -
Before computing the threshold/missing direction, store the indices of the missing values at the end. (Do this before sorting, to avoid additional swaps.)
Notify the criterion that last n_missing samples are missing (init_missing(n_missing)), so it can compute their statistics beforehand and cache it. Also set the initial direction to right.
Iterate from start till end - n_missing, compute all possible splits with missing sent right.
Change the direction (simply adjust the right/left part statistics based on the previously computed statistics of missing values. No need to move the missing samples).
Iterate from start till end-n_missing, compute all possible splits.
Consider the split where missing values are sent to one partition and the available values to the other.
Get the best of all such splits.
Do this for all the non-constant features
Find the best of all such best splits, now we have final best's threshold, missing_direction, n_missing. Lets repartition the samples based on this best split.
. Move the missing to the start if best's missing_direction is left or to the end otherwise.
Repartition the available samples based on best's threshold.
Reset the criterion (This time assume there are no missing samples as we have sent it to the correct partition.)
Recompute the split statistics for the best feature at this new position.
Get the final children's impurity
```
