"""Register file energy study: CACTI sweep plus all model experiments.

Run from inside the compiled CACTI folder (the one containing ./cacti and cache.cfg):
    python rf_energy_experiments.py sweep        # 1,500 CACTI runs -> rf_sweep.csv
    python rf_energy_experiments.py powerlaw     # scaling exponents -> powerlaw.csv
    python rf_energy_experiments.py test1        # random 80/20 splits -> test1.csv
    python rf_energy_experiments.py test2        # leave one technology node out -> test2.csv
    python rf_energy_experiments.py fewshot      # k designs from the new node -> fewshot.csv
    python rf_energy_experiments.py proxy        # device level leakage estimate -> leak_proxy.csv, test2_proxy.csv
    python rf_energy_experiments.py all          # everything, in order

Every error is reported as median, 90th and 98th percentile absolute percentage error.
CACTI and all models use fixed seeds, so reruns reproduce the same numbers.
"""
import itertools
import os
import re
import subprocess
import sys
import tempfile
import time
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

TARGETS = ["read_nj", "write_nj", "leak_mw", "area_mm2", "access_ns"]

# ----------------------------------------------------------------------------
# CACTI sweep
# ----------------------------------------------------------------------------
DESIGN_SPACE = dict(
    entries=[32, 64, 128, 256, 512],            # number of registers
    width=[32, 64, 128],                        # bits per register
    rports=[1, 2, 4, 6, 8],                     # read ports
    wports=[1, 2, 3, 4],                        # write ports
    tech=[0.090, 0.065, 0.045, 0.032, 0.022],   # technology node in microns
)

PATTERNS = {
    "access_ns": r"Access time \(ns\): ([\d.eE+-]+)",
    "read_nj": r"Total dynamic read energy per access \(nJ\): ([\d.eE+-]+)",
    "write_nj": r"Total dynamic write energy per access \(nJ\): ([\d.eE+-]+)",
    "leak_mw": r"Total leakage power of a bank \(mW\): ([\d.eE+-]+)",
    "area_mm2": r"Data array: Area \(mm2\): ([\d.eE+-]+)",
}


def make_cfg(base, entries, width, rports, wports, tech):
    subs = {
        r"^-size \(bytes\) .*$": f"-size (bytes) {entries * width // 8}",
        r"^-block size \(bytes\) .*$": f"-block size (bytes) {width // 8}",
        r"^-associativity .*$": "-associativity 1",
        r"^-read-write port .*$": "-read-write port 0",
        r"^-exclusive read port .*$": f"-exclusive read port {rports}",
        r"^-exclusive write port .*$": f"-exclusive write port {wports}",
        r"^-cache type .*$": '-cache type "ram"',
        r"^-output/input bus width .*$": f"-output/input bus width {width}",
        r"^-technology \(u\) .*$": f"-technology (u) {tech}",
    }
    cfg = base
    for pat, rep in subs.items():
        cfg = re.sub(pat, rep, cfg, count=1, flags=re.M)
    return cfg


def run_cacti(design, base, cacti):
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "rf.cfg")
        with open(path, "w") as f:
            f.write(make_cfg(base, *design))
        try:
            # CACTI reads its technology files relative to its own folder
            out = subprocess.run([cacti, "-infile", path], cwd=os.path.dirname(cacti),
                                 capture_output=True, text=True, timeout=120).stdout
        except subprocess.TimeoutExpired:
            out = ""
    entries, width, rports, wports, tech = design
    row = dict(entries=entries, width=width, rports=rports, wports=wports, tech_nm=round(tech * 1000))
    for key, pat in PATTERNS.items():
        m = re.search(pat, out)
        row[key] = float(m.group(1)) if m else None
    return row


def sweep():
    base = open("cache.cfg").read()
    cacti = os.path.abspath("cacti")
    designs = list(itertools.product(*DESIGN_SPACE.values()))
    t = time.time()
    with ThreadPoolExecutor(max_workers=4) as ex:
        rows = list(ex.map(lambda d: run_cacti(d, base, cacti), designs))
    df = pd.DataFrame(rows)
    df.to_csv("rf_sweep.csv", index=False)
    print(f"{len(df)} designs in {time.time() - t:.0f}s, failed: {df.read_nj.isna().sum()}")


# ----------------------------------------------------------------------------
# Models and metrics
# ----------------------------------------------------------------------------
def feats(d):
    return pd.DataFrame({"log_entries": np.log2(d.entries), "log_width": np.log2(d.width),
                         "rports": d.rports, "wports": d.wports, "log_tech": np.log(d.tech_nm)})


def errors(y_true, log_pred):
    """Median, 90th and 98th percentile absolute percentage error."""
    e = np.abs(np.exp(log_pred) / np.asarray(y_true) - 1) * 100
    return np.median(e), np.percentile(e, 90), np.percentile(e, 98)


class Hybrid:
    """Power law for the overall trend, gradient boosting on its residuals without tech node.

    In linear space this is power_law(x) * exp(g(x without tech)), a multiplicative
    analytical times learned form in the spirit of PANDA.
    """

    def fit(self, X, y):
        self.lin = LinearRegression().fit(X, y)
        self.gb = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, random_state=0)
        self.gb.fit(X.drop(columns="log_tech"), y - self.lin.predict(X))
        return self

    def predict(self, X):
        return self.lin.predict(X) + self.gb.predict(X.drop(columns="log_tech"))


def power_law():
    return LinearRegression()


def random_forest():
    return RandomForestRegressor(n_estimators=300, random_state=0)


def grad_boosting(min_samples_leaf=20):
    return HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05,
                                         min_samples_leaf=min_samples_leaf, random_state=0)


def mlp(seed=0):
    return make_pipeline(StandardScaler(),
                         MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=3000, random_state=seed))


def fit_eval(model, tr, te, target):
    model.fit(feats(tr), np.log(tr[target]))
    return errors(te[target], model.predict(feats(te)))


def load():
    df = pd.read_csv("rf_sweep.csv")
    assert df[TARGETS].notna().all().all(), "sweep has failed designs"
    return df


# ----------------------------------------------------------------------------
# Experiments
# ----------------------------------------------------------------------------
def powerlaw_exponents():
    df = load()
    X = np.log(pd.DataFrame({"entries": df.entries, "width": df.width,
                             "ports": df.rports + df.wports, "tech": df.tech_nm})).assign(const=1.0)
    rows = []
    for t in TARGETS:
        y = np.log(df[t])
        coef, *_ = np.linalg.lstsq(X.values, y.values, rcond=None)
        pred = X.values @ coef
        r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        md, p90, p98 = errors(df[t], pred)
        rows.append(dict(target=t, entries=coef[0], width=coef[1], ports=coef[2], tech=coef[3],
                         R2=r2, median=md, p90=p90, p98=p98))
    out = pd.DataFrame(rows).round(3)
    out.to_csv("powerlaw.csv", index=False)
    print(out.to_string(index=False))


def test1():
    """Random 80/20 splits, 5 seeds."""
    df = load()
    rows = []
    for t in TARGETS:
        for seed in range(5):
            tr, te = train_test_split(df, test_size=0.2, random_state=seed)
            models = {"power law": power_law(), "random forest": random_forest(),
                      "grad boosting": grad_boosting(), "MLP": mlp(seed), "hybrid": Hybrid()}
            for name, m in models.items():
                md, p90, p98 = fit_eval(m, tr, te, t)
                rows.append(dict(target=t, seed=seed, model=name, median=md, p90=p90, p98=p98))
    out = pd.DataFrame(rows)
    out.to_csv("test1.csv", index=False)
    print(out.groupby(["target", "model"])[["median", "p90", "p98"]].mean().round(1).to_string())


def test2():
    """Leave one technology node out. MLP averaged over 5 seeds."""
    df = load()
    rows = []
    for t in TARGETS:
        for node in sorted(df.tech_nm.unique()):
            tr, te = df[df.tech_nm != node], df[df.tech_nm == node]
            groups = {"power law": [power_law()], "random forest": [random_forest()],
                      "grad boosting": [grad_boosting()], "hybrid": [Hybrid()],
                      "MLP": [mlp(s) for s in range(5)]}
            for name, ms in groups.items():
                vals = np.array([fit_eval(m, tr, te, t) for m in ms])
                md, p90, p98 = vals.mean(axis=0)
                rows.append(dict(target=t, node=node, model=name, median=md, p90=p90, p98=p98,
                                 median_sd=vals[:, 0].std()))
    out = pd.DataFrame(rows)
    out.to_csv("test2.csv", index=False)
    for col in ["median", "p98"]:
        print(f"\n{col}")
        print(out.pivot_table(index=["target", "model"], columns="node", values=col).round(1).to_string())


def fewshot(targets=("leak_mw", "read_nj"), nodes=(22, 65, 90), ks=(0, 5, 10, 25, 50), reps=10, mlp_reps=3):
    """Leave one node out, then move k random designs from that node into training."""
    df = load()
    rows = []
    for t in targets:
        for node in nodes:
            other, held = df[df.tech_nm != node], df[df.tech_nm == node]
            for k in ks:
                for rep in range(reps if k else 1):
                    idx = np.random.RandomState(rep).choice(len(held), size=k, replace=False) if k else []
                    tr = pd.concat([other, held.iloc[idx]])
                    te = held.drop(held.index[idx])  # never test on a design used for training
                    models = {"power law": power_law(), "grad boosting": grad_boosting(),
                              "grad boosting leaf5": grad_boosting(min_samples_leaf=5), "hybrid": Hybrid()}
                    if rep < mlp_reps:
                        models["MLP"] = mlp(0)
                    for name, m in models.items():
                        md, p90, p98 = fit_eval(m, tr, te, t)
                        rows.append(dict(target=t, node=node, k=k, rep=rep, model=name,
                                         median=md, p90=p90, p98=p98))
            print(f"done: {t} at {node} nm", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv("fewshot.csv", index=False)
    g = out.groupby(["target", "node", "model", "k"])["median"]
    summary = g.mean().round(1).astype(str) + " ±" + g.std().fillna(0).round(1).astype(str)
    print(summary.unstack("k").to_string())


def device_leak_proxy(node_nm, temp_row=60):
    """Off current (itrs-hp, at 300 K + temp_row, matching the 360 K config) x node size x Vdd,
    read from CACTI's own technology files."""
    with open(os.path.join("tech_params", f"{node_nm}nm.dat")) as f:
        s = f.read()
    vdd = float(re.search(r"^-Vdd \(V\) ([\d.]+)", s, re.M).group(1))
    ioff = float(re.search(rf"^-I_off_n \(A/um\) {temp_row} ([\d.eE+-]+)", s, re.M).group(1))
    return ioff * node_nm * vdd, ioff, vdd


class HybridProxy(Hybrid):
    """Hybrid whose power law also sees the device leakage estimate; trees see design features only."""
    DROP = ["log_tech", "log_leak_proxy"]

    def fit(self, X, y):
        self.lin = LinearRegression().fit(X, y)
        self.gb = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, random_state=0)
        self.gb.fit(X.drop(columns=self.DROP), y - self.lin.predict(X))
        return self

    def predict(self, X):
        return self.lin.predict(X) + self.gb.predict(X.drop(columns=self.DROP))


def proxy():
    """Unseen node test with a device level leakage estimate from CACTI's technology files."""
    df = load()
    nodes = [90, 65, 45, 32, 22]
    table = pd.DataFrame({n: dict(zip(["proxy", "I_off_A_per_um", "Vdd"], device_leak_proxy(n)))
                          for n in nodes}).T
    table["proxy_rel"] = table.proxy / table.proxy[90]
    med = df.groupby("tech_nm").leak_mw.median()
    table["cacti_leak_median_rel"] = med / med[90]
    table.to_csv("leak_proxy.csv")
    print(table.to_string(float_format=lambda v: f"{v:.3g}"), "\n")

    rel = table.proxy_rel.to_dict()

    def feats_p(d):
        f = feats(d)
        f["log_leak_proxy"] = np.log(d.tech_nm.map(rel))
        return f

    rows = []
    for t in TARGETS:
        for node in sorted(df.tech_nm.unique()):
            tr, te = df[df.tech_nm != node], df[df.tech_nm == node]
            for name, m, fx in [("hybrid", Hybrid(), feats), ("hybrid + proxy", HybridProxy(), feats_p),
                                ("power law + proxy", power_law(), feats_p)]:
                m.fit(fx(tr), np.log(tr[t]))
                md, p90, p98 = errors(te[t], m.predict(fx(te)))
                rows.append(dict(target=t, node=node, model=name, median=md, p90=p90, p98=p98))
    out = pd.DataFrame(rows)
    out.to_csv("test2_proxy.csv", index=False)
    print(out.pivot_table(index=["target", "model"], columns="node", values="median").round(1).to_string())


STEPS = {"sweep": sweep, "powerlaw": powerlaw_exponents, "test1": test1, "test2": test2, "fewshot": fewshot, "proxy": proxy}

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name in (STEPS if which == "all" else [which]):
        print(f"\n===== {name} =====")
        STEPS[name]()
