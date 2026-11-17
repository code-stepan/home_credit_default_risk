from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-6
MISSING_BIN = "__MISSING__"


class WOETransformer:
    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins
        self.edges_:dict[str, np.ndarray] = {}
        self.woe_maps_:dict[str, dict] = {}
        self.iv_: dict[str, float] = {}
        self._numeric_cols_: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "WOETransformer":
        y = pd.Series(y).reset_index(drop=True)
        total_good = float((y == 0).sum())
        total_bad = float((y == 1).sum())
        self._numeric_cols_ = [
            c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])
        ]
        for col in X.columns:
            binned = self._to_bins(X[col], col)
            woe, iv = self._woe_from_bins(binned, y, total_good, total_bad)
            self.woe_maps_[col] = woe
            self.iv_[col] = iv
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = {}
        for col in X.columns:
            binned = self._to_bins(X[col], col)
            # woepath
            out[col] = binned.map(pd.Series(self.woe_maps_[col])).astype(float)
        return pd.DataFrame(out, index=X.index).fillna(0.0)

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    def _to_bins(self, s: pd.Series, col: str) -> pd.Series:
        if col in self._numeric_cols_:
            edges = self.edges_.get(col)
            if edges is None:
                q = np.linspace(0, 1, self.n_bins + 1)
                edges = np.unique(s.dropna().quantile(q).values)
                self.edges_[col] = edges
            if len(edges) < 2:
                return pd.Series("bin_0", index=s.index, dtype=object)
            clipped = s.clip(edges[0], edges[-1])
            b = pd.cut(clipped, bins=edges, labels=False, include_lowest=True)
            return b.map(lambda v: MISSING_BIN if pd.isna(v) else f"bin_{int(v)}")
        return s.fillna(MISSING_BIN).astype(str)

    @staticmethod
    def _woe_from_bins(binned, y, total_good, total_bad):
        df = pd.DataFrame({"bin": binned.reset_index(drop=True), "y": y})
        agg = df.groupby("bin")["y"].agg(["count", "sum"])
        good = agg["count"] - agg["sum"]
        bad = agg["sum"]
        # сглаживание Лапласа
        good_rate = (good + 0.5) / (total_good + 0.5 * len(agg))
        bad_rate = (bad + 0.5) / (total_bad + 0.5 * len(agg))
        woe = np.log((good_rate + EPS) / (bad_rate + EPS))
        iv = float(((good_rate - bad_rate) * woe).sum())
        return woe.to_dict(), iv