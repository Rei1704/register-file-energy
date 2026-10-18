# Register File Energy Across Technology Nodes

A controlled study of how register file dynamic energy, leakage, area and access time scale with design parameters and technology node, and of when machine learning models trained on some technology nodes generalize to a node they have never seen.

All data comes from [CACTI](https://github.com/HewlettPackard/cacti): 1,500 register file designs covering 32 to 512 entries, 32 to 128 bit width, 1 to 8 read ports, 1 to 4 write ports, and five technology nodes from 90 nm to 22 nm. Results reflect CACTI's technology models, not measurements of fabricated chips.

## Main findings

**Random split accuracy is misleading.** Under a standard random 80/20 split, gradient boosting is the most accurate model (1 to 4% median error). On a technology node held out entirely, the same model is off by 45 to 108% on dynamic energy and area, because tree models produce step functions and cannot interpolate between the five technology nodes.

**A multiplicative hybrid generalizes to unseen nodes.** Fitting a power law for the overall trend, then learning its residuals with gradient boosting while keeping technology node out of the learned part, predicts unseen nodes within 1.0 to 4.5% median error for read energy, write energy and area, with 98th percentile errors mostly under 22%.

**Leakage breaks that assumption.** Within CACTI's models, leakage rises from 90 nm to 65 nm and then falls through 22 nm. This reversal is not separable from design effects, so no model trained on other nodes predicts 65 or 90 nm leakage well.

**A few designs from the new node help, but tails remain.** Adding about 25 designs from the held out node brings gradient boosting's median leakage error to 5 to 11%, though 98th percentile errors stay at roughly 25 to 50%. Power law based models do not improve with added samples.

## Reproducing the results

```
git clone https://github.com/HewlettPackard/cacti.git
cd cacti
make
# copy rf_energy_experiments.py into this folder, then:
python rf_energy_experiments.py all
```

CACTI and all models use fixed seeds, so a rerun reproduces the included CSV files exactly. The full run takes about 15 to 20 minutes on a laptop CPU. Individual steps can be run with `sweep`, `powerlaw`, `test1`, `test2` or `fewshot` in place of `all`.

## Files

| File | Contents |
|------|----------|
| `rf_energy_experiments.py` | CACTI sweep and every model experiment |
| `rf_sweep.csv` | The 1,500 design dataset |
| `test2.csv` | Leave one technology node out results, median and tail errors |
| `fewshot.csv` | Results when k designs from the held out node are added to training |

## Related work

This study builds on hybrid analytical and machine learning power models such as PANDA, on analytical frameworks such as ReadyPower, and on low data transfer learning for embedded memory PPA models (Last and Schlichtmann, ACM TODAES 2022).

A preprint describing this work is in preparation.
