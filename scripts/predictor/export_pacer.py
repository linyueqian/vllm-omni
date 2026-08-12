"""Export the pacing brain: deterministically retrain the blocked-split
gbm-q0.9 (log target, seed/params identical to the committed predictor run)
and joblib-dump {model, conformal offset Qc, ordered feature list}.

Run on the machine whose venv will LOAD the model (sklearn pickle
compatibility): .venv/bin/python scripts/predictor/export_pacer.py
"""
import os

import joblib
import numpy as np
import sklearn

from dstar_predictor import (FEATURES, alloc_blocked, conformal_offset,
                             fit_gbm, load, masks, run_table)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pacer_model.joblib")
LOG_TARGET = True  # frozen choice from the committed blocked-split run


def main() -> None:
    X, y, rates, run_ids, t = load()
    runs = run_table(run_ids, rates, t)
    alloc = alloc_blocked(runs)
    m = masks(alloc, run_ids)
    fpred, model = fit_gbm(X[m["train"]], y[m["train"]], "quantile", LOG_TARGET)
    Qc = conformal_offset(y[m["cal"]], fpred(X[m["cal"]]))

    # Reproducibility fingerprint: Qc and a few fixed-row predictions.
    probe = fpred(X[:5])
    payload = {
        "model": model,
        "Qc": float(Qc),
        "features": list(FEATURES),
        "log_target": LOG_TARGET,
        "sklearn_version": sklearn.__version__,
        "fingerprint": {
            "Qc": round(float(Qc), 6),
            "pred_rows_0_4": [round(float(p), 6) for p in probe],
            "n_train": int(m["train"].sum()),
            "n_cal": int(m["cal"].sum()),
        },
    }
    joblib.dump(payload, OUT)
    print(f"wrote {OUT}")
    print(f"sklearn {sklearn.__version__}  Qc={Qc:.4f}  "
          f"n_train={m['train'].sum()}  n_cal={m['cal'].sum()}")
    print(f"pred rows 0-4: {np.round(probe, 4).tolist()}")


if __name__ == "__main__":
    main()
