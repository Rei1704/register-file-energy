# Register File Energy Across Technology Nodes

A controlled study of how register file dynamic energy, leakage, area and access time scale with design parameters and technology node, and of when machine learning models trained on some technology nodes generalize to a node they have never seen.

All data comes from [CACTI](https://github.com/HewlettPackard/cacti): 1,500 register file designs covering 32 to 512 entries, 32 to 128 bit width, 1 to 8 read ports, 1 to 4 write ports, and five technology nodes from 90 nm to 22 nm, using high performance (itrs-hp) devices at 360 K. Results reflect CACTI's technology models, not measurements of fabricated chips.

## Main findings

**Random split accuracy is misleading.** Under a standard random 80/20 split, gradient boosting is the most accurate model (1.2 to 3.6% median error). On a technology node held out entirely, the same model is off by 45 to 108% on dynamic energy and area. Tree models produce step functions, so instead of interpolating between the five technology nodes they predict a neighboring node's values (Figure 4).

**A multiplicative hybrid generalizes to unseen nodes.** Fitting a power law for the overall trend, then learning its residuals with gradient boosting while keeping technology node out of the learned part, predicts unseen nodes within 1.0 to 4.5% median error for read energy, write energy and area, with 98th percentile errors of 6 to 22%.

**Leakage is non monotonic across nodes, and the cause is visible in CACTI's device parameters.** Median leakage rises about 3x from 90 nm to 65 nm and then falls through 22 nm (Figure 1). Multiplying CACTI's off current per micrometer by feature size and supply voltage reproduces this trend within about 7% at every node: off current rises faster than widths and voltage shrink until 45 nm, after which the shrinking wins.

**With that device level estimate, leakage becomes predictable on unseen nodes.** Adding it as an input to the hybrid's power law brings unseen node leakage error from 50 to 450% down to 2.3 to 6.2% for 22 to 65 nm, and 26% at 90 nm, with no designs from the new node. The estimate helps only leakage; for the other outputs the plain hybrid remains best. Because CACTI computes leakage from these same parameters, the relationship is close to exact here and would be approximate on real silicon.

**Without device parameters, a few new node designs help, but tails remain.** Adding about 25 designs from the held out node brings gradient boosting's median leakage error to 5 to 11%, though 98th percentile errors stay at roughly 25 to 50% (Figure 3). Power law based models do not improve with added samples.

## Reproducing the results

```
git clone https://github.com/HewlettPackard/cacti.git
cd cacti
make
# copy rf_energy_experiments.py into this folder, then:
python rf_energy_experiments.py all
```

CACTI and all models use fixed seeds, so a rerun reproduces the included CSV files exactly. The full run takes about 15 to 20 minutes on a laptop CPU. Individual steps can be run with `sweep`, `powerlaw`, `test1`, `test2`, `fewshot` or `proxy` in place of `all`. The `proxy` step reads CACTI's `tech_params` files, so it must also be run from inside the CACTI folder.

## Files

| File | Contents |
|------|----------|
| `rf_energy_experiments.py` | CACTI sweep and every experiment |
| `rf_sweep.csv` | The 1,500 design dataset |
| `test1.csv` | Random 80/20 split results, 5 seeds |
| `test2.csv` | Leave one technology node out results |
| `fewshot.csv` | Results when k designs from the held out node are added to training |
| `leak_proxy.csv` | Device level leakage estimate per node from CACTI's technology files |
| `test2_proxy.csv` | Leave one node out results with the device level estimate |
| `fig1_scaling.png` | Median outputs per node, relative to 90 nm |
| `fig2_split_vs_node.png` | Random split versus unseen node error for each model |
| `fig3_fewshot.png` | Leakage error versus number of new node designs |
| `fig4_parity.png` | Predicted versus CACTI read energy at an unseen node |

All errors are reported as median, 90th and 98th percentile absolute percentage error.

## Related work

This study builds on hybrid analytical and machine learning power models such as PANDA, on analytical frameworks such as ReadyPower, and on low data transfer learning for embedded memory PPA models (Last and Schlichtmann, ACM TODAES 2022).

A preprint describing this work is in preparation.
