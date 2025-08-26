from __future__ import annotations
import os, math, pickle, warnings, datetime as _dt, logging, random, hashlib
from pathlib import Path
from contextlib import contextmanager
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf
from pandas_datareader.data import DataReader
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from pykrx import stock
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge, Lasso, ElasticNet, HuberRegressor
from sklearn.pipeline import Pipeline

import plotly.graph_objects as go

# ─────────────────── 0. 옵션 & 전역 상수 ─────────────────── #
SEED = 42
np.random.seed(SEED); random.seed(SEED)

START_DATE = "1999-01-01"
END_DATE = _dt.date.today().strftime("%Y-%m-%d")
END_PD = pd.to_datetime(END_DATE)

CACHE_DIR = Path(".cache"); CACHE_DIR.mkdir(exist_ok=True)
LOG_FILE = "kospi_risk_score_v8.log"

# 튜닝/모델
OPTUNA_TRIALS = int(os.getenv("N_OPTUNA_TRIALS", 30))
CV_FOLDS = 6
MIN_DATA_FOR_TRAIN = 252 * 5
MODEL_TYPES = ["lasso", "ridge", "elastic", "huber"]
ALPHA_MIN, ALPHA_MAX= 0.1, 50.0 # [MODIFIED v8.8] 규제 강화
STEP_CAND = [5]
ROLL_WIN_CAND = [252, 504]
CORR_WIN_CAND = [126, 252, 504]
FUTURE_DAYS_ENSEMBLE= [63, 126, 252]

# 배깅/스무딩
BAGS = 5
BAG_SAMPLE_FRAC = 0.8
BLOCK_SIZE = 20
SMOOTH_LAMBDA_CAND = [0.7, 0.8, 0.85, 0.9]

# 스케일/최종 스케일
SCALING_METHOD = "percentile" # ("percentile", "z", "none")
FINAL_RISK_SCALER = "minmax" # ("minmax","percentile","none")

# 적응형 앙상블(누수 없는 방식) — 기본 활성화
ADAPTIVE_ENSEMBLE = True
ADAPTIVE_LOOKBACK = 252 * 3 # 3년

YF_TICKERS: Dict[str, str] = {
    "KOSPI":"^KS11","SP500":"^GSPC","VIX":"^VIX","USDKRW":"KRW=X",
    "CRUDE_OIL":"CL=F","COPPER":"HG=F","GOLD":"GC=F","IEF":"IEF",
}
INCEPTION_DATES = {"IEF":"2002-07-22"}

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ─────────────────── 로깅 ─────────────────── #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, mode='w', encoding="utf-8"),
              logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# ─────────────────── 1. 헬퍼 ─────────────────── #
@contextmanager
def _open_pickle(path: Path, mode: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, mode) as f:
        yield f

def _save(obj: Any, path: Path)->None:
    with _open_pickle(path, "wb") as f:
        pickle.dump(obj, f)

def _load(path: Path)->Any|None:
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)

def _date_range(s: pd.Series|pd.DataFrame):
    d = s.dropna()
    return (d.index.min(), d.index.max()) if not d.empty else (None, None)

def safe_spearman(x: pd.Series, y: pd.Series)->Tuple[float,float]:
    """NaN-safe Spearman; 표본/고유값 부족 시 NaN."""
    x = pd.Series(x).astype(float)
    y = pd.Series(y).astype(float)
    df = pd.concat([x, y], axis=1).dropna()
    if df.shape[0] < 3 or df.iloc[:,0].nunique() < 2 or df.iloc[:,1].nunique() < 2:
        return (np.nan, np.nan)
    return spearmanr(df.iloc[:,0], df.iloc[:,1])

def rolling_percentile(s: pd.Series, window:int)->pd.Series:
    arr = s.to_numpy(float); out = np.full_like(arr, np.nan, float)
    for i in range(len(arr)):
        if math.isnan(arr[i]): continue
        lo = max(0, i - window); win = arr[lo:i]
        win = win[~np.isnan(win)]
        if win.size >= max(10, window - 1):
            rank = (win <= arr[i]).mean() * 100.0
            out[i] = rank
    return pd.Series(out, index=s.index)

def rolling_minmax(s: pd.Series, window:int)->pd.Series:
    rmin = s.rolling(window, min_periods=window).min()
    rmax = s.rolling(window, min_periods=window).max()
    denom = (rmax - rmin)
    out = np.where(denom == 0, 50.0, (s - rmin) / denom * 100.0)
    return pd.Series(out, index=s.index).clip(0, 100)

def rolling_zscore(s: pd.Series, window:int)->pd.Series:
    m = s.rolling(window, min_periods=window).mean()
    v = s.rolling(window, min_periods=window).std()
    r = (s - m) / v
    return r.replace([np.inf, -np.inf], np.nan)

def rank_quantile_labels(series: pd.Series, q:int=5)->pd.Series:
    pct = series.rank(pct=True, method="average")
    labels = np.ceil(pct * q).astype(int)
    labels = labels.clip(1, q)
    return pd.Series(labels, index=series.index)

def winsorize_series(s: pd.Series, p:float=0.01)->pd.Series:
    lo, hi = s.quantile(p), s.quantile(1-p)
    return s.clip(lower=lo, upper=hi)

def block_bootstrap_ic(risk: pd.Series, y: pd.Series, horizon:int, block:int=20, n:int=1000, seed:int=SEED)\
        ->Tuple[float, Tuple[float,float]]:
    rng = np.random.default_rng(seed)
    target = y.pct_change(horizon).shift(-horizon)
    common = risk.dropna().index.intersection(target.dropna().index)
    r = risk.loc[common].to_numpy(); t = target.loc[common].to_numpy()
    n_obs = len(common)
    if n_obs < block*2:
        return (np.nan, (np.nan, np.nan))
    ics = []
    for _ in range(n):
        idx = []
        while len(idx) < n_obs:
            start = rng.integers(0, max(1, n_obs - block))
            idx.extend(range(start, start + block))
        idx = np.array(idx[:n_obs])
        ic, _ = spearmanr(r[idx], t[idx])
        if not np.isnan(ic): ics.append(ic)
    arr = np.array(ics)
    return (arr.mean(), (np.quantile(arr, 0.025), np.quantile(arr, 0.975)))

# ─────────────────── 2. 데이터 ─────────────────── #
def _get_cached(name:str, loader)->pd.DataFrame:
    cache = CACHE_DIR / f"{name}.pkl"
    data = _load(cache)
    if isinstance(data, pd.DataFrame) and not data.empty and data.index.max() >= END_PD - pd.Timedelta(days=7):
        logger.info(f"{name}: Using cached data.")
        return data
    logger.info(f"{name}: Downloading data.")
    data = loader(); _save(data, cache); return data

def get_price_data()->pd.DataFrame:
    def _dl():
        df = pd.DataFrame()
        for k, t in YF_TICKERS.items():
            logger.info(f"  ▸ Downloading {k} ({t}) ...")
            s = yf.download(t, start=START_DATE, end=END_DATE, auto_adjust=True, progress=False)["Close"]
            if k in INCEPTION_DATES:
                s = s.loc[s.index >= pd.to_datetime(INCEPTION_DATES[k])]
            df[k] = s
        return df.ffill()
    df = _get_cached("price_data", _dl)
    if "IEF" in INCEPTION_DATES:
        df.loc[df.index < pd.to_datetime(INCEPTION_DATES["IEF"]), "IEF"] = np.nan
    return df.loc[:END_PD]

def get_yields()->pd.DataFrame:
    def _dl():
        logger.info("Downloading Yields from FRED: DGS10 & DGS2 ...")
        d10 = DataReader("DGS10", "fred", START_DATE, END_DATE)
        d02 = DataReader("DGS2",  "fred", START_DATE, END_DATE)
        y = pd.concat({"US10Y": d10.squeeze(), "US2Y": d02.squeeze()}, axis=1)
        return y
    y = _get_cached("yields_fred", _dl).loc[:END_PD]
    return y.ffill()

def get_per_pbr()->pd.DataFrame:
    def _dl():
        logger.info("Downloading PER/PBR (KOSPI index fundamentals)...")
        d = stock.get_index_fundamental("19990104", END_PD.strftime("%Y%m%d"), "1001").rename_axis("Date").reset_index()
        d["Date"] = pd.to_datetime(d["Date"])
        return d[["Date", "PER", "PBR"]].set_index("Date").replace(0, np.nan).ffill()
    return _get_cached("per_pbr", _dl).loc[:END_PD]

def _build_flow_norm_from_raw(df_raw: pd.DataFrame)->pd.DataFrame:
    col_map = {}
    if "개인" in df_raw.columns: col_map["개인"] = "Individual"
    if "기관합계" in df_raw.columns: col_map["기관합계"] = "Institution"
    if "외국인합계" in df_raw.columns: col_map["외국인합계"] = "Foreign"
    if "전체" in df_raw.columns: col_map["전체"] = "Total"

    df = df_raw.rename(columns=col_map).copy()
    needed = ["Individual","Institution","Foreign"]
    for c in needed:
        if c not in df.columns:
            raise KeyError(f"Missing column in pykrx data: {c}")

    for col in set(needed + (["Total"] if "Total" in df.columns else [])):
        df[f"{col}_20D"] = df[col].rolling(20, min_periods=20).sum()

    if "Total_20D" in df.columns:
        base = df["Total_20D"].abs()
    else:
        logger.warning("pykrx data has no '전체' column. Approximating Total_20D by sum of parties.")
        base = (df["Individual_20D"].abs() + df["Institution_20D"].abs() + df["Foreign_20D"].abs())

    eps = 1e-12
    out = pd.DataFrame(index=df.index)
    out["F_FlowR"] = df["Foreign_20D"] / (base + eps)
    out["I_FlowR"] = df["Institution_20D"] / (base + eps)
    out["D_FlowR"] = df["Individual_20D"] / (base + eps)

    out = out.apply(lambda s: winsorize_series(s, 0.01)).clip(0, 1)
    return out.replace([np.inf, -np.inf], np.nan).ffill()

def get_flow_norm()->pd.DataFrame:
    def _dl():
        logger.info("Downloading flow (trading value by investor type)...")
        d = stock.get_market_trading_value_by_date("19990104", END_DATE.replace("-", ""), "KOSPI")\
                 .rename_axis("Date").reset_index()
        d["Date"] = pd.to_datetime(d["Date"])
        d = d.set_index("Date").ffill()
        return _build_flow_norm_from_raw(d)
    return _get_cached("flow_norm", _dl).loc[:END_PD]

# ─────────────────── 3. 특징 계산 (+ 캐시) ─────────────────── #
_FEATURES_CACHE: Dict[str, pd.DataFrame] = {}

def _features_cache_key(kospi_idx: pd.Index, roll_window:int)->str:
    h = hashlib.md5((",".join(str(x) for x in kospi_idx[:10]) + "|" + str(len(kospi_idx))).encode()).hexdigest()
    return f"{h}|roll={roll_window}|scale={SCALING_METHOD}"

def get_dynamic_directions(features_df: pd.DataFrame, y: pd.Series, horizon: int) -> List[str]:
    """
    [NEW v8.8]
    Calculate the correlation of each feature with future returns over a given period.
    Returns a list of features that have a POSITIVE correlation, indicating they
    should be inverted to be treated as risk factors.
    """
    target = y.pct_change(horizon).shift(-horizon)
    positive_corr_features = []
    logger.info(f"Calculating dynamic directions over {y.index.min().year}-{y.index.max().year} for H={horizon}...")
    for col in features_df.columns:
        ic, p_val = safe_spearman(features_df[col], target)
        if isinstance(ic, float) and not np.isnan(ic) and ic > 0:
            positive_corr_features.append(col)
            logger.info(f"  ▸ Feature '{col}' has positive correlation ({ic:.3f}). Will be inverted.")
    return positive_corr_features

def _calculate_features(
    kospi: pd.Series, prices: pd.DataFrame, yields: pd.DataFrame, per_pbr: pd.DataFrame,
    flows_norm: pd.DataFrame, roll_window: int, features_to_flip: List[str]
)->pd.DataFrame:
    """
    [MODIFIED v8.8]
    - Accepts `features_to_flip`, a list of feature names.
    - This list is determined by `get_dynamic_directions` based on tuning data.
    - It inverts the scale for these specific features to align them as risk factors.
    """
    idx = kospi.index
    for x in (prices, yields, per_pbr, flows_norm):
        idx = idx.intersection(x.index)
    kospi, prices, yields, per_pbr, flows = [x.reindex(idx).ffill() for x in (kospi, prices, yields, per_pbr, flows_norm)]

    key = _features_cache_key(kospi.index, roll_window)
    if key in _FEATURES_CACHE and key+"|"+",".join(features_to_flip) in _FEATURES_CACHE:
         return _FEATURES_CACHE[key+"|"+",".join(features_to_flip)].copy()

    yld_spr = (yields["US10Y"] - yields["US2Y"]).rename("YldSpr")

    df = pd.DataFrame({
        "Mom20":      kospi.pct_change(20), "MA50_Rel":   kospi / kospi.rolling(50, min_periods=50).mean() - 1.0,
        "Vol20":      kospi.pct_change().rolling(20, min_periods=20).std() * np.sqrt(252),
        "VolRatio":   (kospi.pct_change().rolling(20, min_periods=20).std() * np.sqrt(252)) /
                      (prices["SP500"].pct_change().rolling(20, min_periods=20).std() * np.sqrt(252)).replace(0, np.nan),
        "Ret20":      kospi.pct_change(20), "RetDiff":    kospi.pct_change(20) - prices["SP500"].pct_change(20),
        "KRW_Change": prices["USDKRW"].pct_change(20), "YldSpr":     yld_spr,
        "Cu_Gd":      (prices["COPPER"] / prices["GOLD"]).replace([np.inf, -np.inf], np.nan),
        "PER":        per_pbr["PER"], "PBR":        per_pbr["PBR"], "VIX":        prices["VIX"],
        "F_FlowR":    flows["F_FlowR"], "I_FlowR":    flows["I_FlowR"], "D_FlowR":    flows["D_FlowR"] ,
        "Oil_Ret":    prices["CRUDE_OIL"].pct_change(20), "IEF_Ret":    prices["IEF"].pct_change(20),
        "IEF_Vol":    prices["IEF"].pct_change().rolling(20, min_periods=20).std() * np.sqrt(252),
    }).ffill()

    # Step 1: Scaling
    if SCALING_METHOD == "percentile":
        scaled = pd.concat({c: rolling_percentile(df[c], roll_window) for c in df.columns}, axis=1).clip(0, 100)
    elif SCALING_METHOD == "z":
        z = df.apply(lambda x: rolling_zscore(x, roll_window))
        scaled = (z.clip(-3, 3) * 10 + 50).clip(0, 100)
    else: # "none"
        scaled = df.copy()

    # Step 2: Apply dynamic direction flip
    for col in features_to_flip:
        if col in scaled.columns:
            if SCALING_METHOD in ("percentile", "z"):
                scaled[col] = 100 - scaled[col]
            else: # 'none'
                scaled[col] = -scaled[col]

    sma200 = kospi.rolling(200, min_periods=200).mean()
    regime = (kospi >= sma200).astype(float).rename("RegimeBull")
    scaled = pd.concat([scaled, regime.reindex(scaled.index)], axis=1)

    scaled = scaled.dropna(how='all')
    _FEATURES_CACHE[key+"|"+",".join(features_to_flip)] = scaled.copy()
    return scaled

# ─────────────────── 4. 모델 학습 유틸 ─────────────────── #
def _get_model(model_type:str, alpha:float):
    if model_type == "lasso": return Lasso(alpha=alpha, max_iter=10000)
    if model_type == "ridge": return Ridge(alpha=alpha)
    if model_type == "elastic": return ElasticNet(alpha=alpha, l1_ratio=0.5, max_iter=10000)
    if model_type == "huber": return HuberRegressor(alpha=alpha)
    raise ValueError("Unknown model_type")

def _fit_with_bagging(X: pd.DataFrame, y: pd.Series, model_type:str, alpha:float, prev_w: Optional[np.ndarray], lam: float)->np.ndarray:
    cols = X.columns; coefs = []; n = len(X); base_model = _get_model(model_type, alpha)
    for _ in range(BAGS):
        want = int(n * BAG_SAMPLE_FRAC); idx_sel = []
        while len(idx_sel) < want:
            start = np.random.randint(0, max(1, n - BLOCK_SIZE))
            idx_sel.extend(range(start, start + BLOCK_SIZE))
        idx_sel = sorted(set(idx_sel[:want])); Xb = X.iloc[idx_sel]; yb = y.iloc[idx_sel]
        pipe = Pipeline([("reg", base_model)])
        try:
            pipe.fit(Xb, yb); coef = np.asarray(pipe.named_steps["reg"].coef_, dtype=float); coefs.append(coef)
        except Exception as e:
            logging.warning(f"Bagging fit skipped due to error: {e}"); continue
    if not coefs: coef_mean = np.zeros(len(cols))
    else:
        coef_mean = np.mean(coefs, axis=0); s = np.sum(np.abs(coef_mean))
        coef_mean = coef_mean / s if s > 1e-12 else np.zeros_like(coef_mean)
    if prev_w is not None and lam is not None:
        coef_mean = lam * prev_w + (1 - lam) * coef_mean; s = np.sum(np.abs(coef_mean))
        coef_mean = coef_mean / s if s > 1e-12 else coef_mean
    return coef_mean

def _train_weights_over_time(
    features: pd.DataFrame, target: pd.Series, correlation_window:int, step:int,
    model_type:str, alpha:float, smooth_lambda:float, embargo:int
)->pd.DataFrame:
    weights = pd.DataFrame(index=features.index, columns=features.columns, dtype=float); prev_w = None
    min_start = correlation_window + embargo
    for i in range(min_start, len(features), step):
        end_idx = i - embargo
        if end_idx <= 0: continue
        start_idx = max(0, end_idx - correlation_window)
        X = features.iloc[start_idx:end_idx]; y = target.iloc[start_idx:end_idx]
        data = pd.concat([X, y.rename("y")], axis=1).dropna()
        if len(data) < int(correlation_window * 0.8): continue
        coef = _fit_with_bagging(data[X.columns], data["y"], model_type, alpha, prev_w, smooth_lambda)
        weights.iloc[i] = coef; prev_w = coef
    return weights.ffill().fillna(0.0)

def _compute_risk_series(features: pd.DataFrame, weights: pd.DataFrame)->pd.Series:
    w = weights.reindex(features.index, method='ffill').fillna(0.0)
    risk = (features * w).sum(axis=1)
    return risk.dropna()

# ─────────────────── 5. CV/튜닝 ─────────────────── #
def _purged_cv_score_single_h(
    features: pd.DataFrame, y: pd.Series, future_days:int, correlation_window:int, step:int,
    model_type:str, alpha:float, smooth_lambda:float, folds:int
)->Tuple[float, List[float]]:
    target = y.pct_change(future_days).shift(-future_days); embargo = future_days; gap = max(future_days, 200)
    total = len(features)
    min_len = max(correlation_window, MIN_DATA_FOR_TRAIN) + future_days + embargo
    if total < min_len: return (np.nan, [])
    first_valid = features.dropna().index[0]; first_pos = features.index.get_loc(first_valid)
    start_idx = first_pos + correlation_window + future_days + embargo
    if start_idx >= total - 22: return (np.nan, [])
    val_len = total - start_idx; fold_size = max(22, val_len // folds)
    scores = []
    for k in range(folds):
        vs = start_idx + k * fold_size; ve = start_idx + (k + 1) * fold_size if k < folds - 1 else total
        train_end_pos = vs - gap
        if train_end_pos <= 0: continue
        train_end_date = features.index[train_end_pos - 1]
        X_tr = features.loc[:train_end_date]; y_tr = y.loc[:train_end_date]
        w = _train_weights_over_time(
            X_tr, y_tr.pct_change(future_days).shift(-future_days),
            correlation_window, step, model_type, alpha, smooth_lambda, embargo=future_days
        )
        idx = features.index[vs:ve]
        risk = _compute_risk_series(features.loc[idx], w.reindex(idx, method='ffill'))
        ret = y.pct_change(future_days).shift(-future_days).loc[idx]
        ic, _ = safe_spearman(risk, ret)
        if isinstance(ic, float) and not np.isnan(ic): scores.append(ic)
    return (np.mean(scores) if scores else np.nan), scores

def _objective(trial, kospi, features_tune_data, y_tune, features_to_flip):
    future_days = trial.suggest_categorical("future_days", FUTURE_DAYS_ENSEMBLE)
    roll_window = trial.suggest_categorical("roll_window", ROLL_WIN_CAND)
    correlation_window = trial.suggest_categorical("correlation_window", CORR_WIN_CAND)
    step = trial.suggest_categorical("step", STEP_CAND)
    model_type = trial.suggest_categorical("model_type", MODEL_TYPES)
    alpha = trial.suggest_float("alpha", ALPHA_MIN, ALPHA_MAX, log=True)
    smooth_lambda = trial.suggest_categorical("smooth_lambda", SMOOTH_LAMBDA_CAND)

    features = _calculate_features(kospi, *features_tune_data, roll_window, features_to_flip)
    score, _ = _purged_cv_score_single_h(features, y_tune, future_days,
                                         correlation_window, step, model_type, alpha, smooth_lambda, CV_FOLDS)
    if score is None or math.isnan(score): return float("inf")
    trial.report(score, 0)
    if trial.should_prune(): raise optuna.TrialPruned()
    logger.info(f"Trial {trial.number} | IC:{score:.4f}")
    return score

# ─────────────────── 6. 리포트 & 진단 ─────────────────── #
def quantile_report(risk: pd.Series, y: pd.Series, horizon:int, q:int=5, label:str="")->pd.DataFrame:
    df = pd.concat({"risk":risk, "y": y.pct_change(horizon).shift(-horizon)}, axis=1).dropna()
    if df.empty: logger.warning(f"[Quantile Report {label}] empty data."); return pd.DataFrame()
    df["q"] = rank_quantile_labels(df["risk"], q=q)
    rep = (df.groupby("q")["y"]
           .agg(n="count", mean_ret="mean", median_ret="median",
                hit_rate_pos=lambda s: (s>0).mean(), hit_rate_neg=lambda s: (s<0).mean()))
    rep.index = [f"Q{int(i)}/{q}" for i in rep.index]
    rep["_qnum"] = np.arange(1, len(rep)+1)
    slope = np.polyfit(rep["_qnum"], rep["mean_ret"], 1)[0] if len(rep)>=2 else np.nan
    logger.info(f"[Quantile Report {label}]\n%s", rep.drop(columns=["_qnum"]).to_string())
    logger.info("Monotonic slope(mean_ret vs q): %.6f (↓가 이상적)", slope if not np.isnan(slope) else np.nan)
    return rep.drop(columns=["_qnum"])

def ic_all_scales(risk: pd.Series, y: pd.Series, horizon:int, label:str="")->Dict[str, float]:
    target = y.pct_change(horizon).shift(-horizon)
    def _ic(series: pd.Series):
        ic, _ = safe_spearman(series, target); return float(ic) if ic==ic else np.nan
    res = {"raw": _ic(risk)}; window_5y = 252*5
    if FINAL_RISK_SCALER == "minmax":
        scaled = rolling_minmax(risk, window_5y).clip(0, 100); res["minmax"] = _ic(scaled)
    if FINAL_RISK_SCALER in ("minmax","percentile"):
        perc = rolling_percentile(risk, window_5y).clip(0, 100); res["percentile"] = _ic(perc)
    logger.info(f"[{label}] IC(raw/minmax/percentile): " + ", ".join([f"{k}:{v:.4f}" for k,v in res.items() if v==v]))
    return res

def ic_decay_curve(risk: pd.Series, y: pd.Series, horizons: List[int])->pd.DataFrame:
    rows=[]
    for h in horizons:
        ic,_ = safe_spearman(risk, y.pct_change(h).shift(-h)); rows.append({"horizon":h, "IC":ic})
    res = pd.DataFrame(rows).dropna()
    if not res.empty: logger.info("[IC Decay]\n%s", res.to_string(index=False))
    return res

# ─────────────────── 7. 앙상블(정적/적응형) ─────────────────── #
def _softmax_abs_ic(ics: Dict[int,float])->Dict[int,float]:
    arr = np.array([abs(ics.get(h,0.0)) for h in FUTURE_DAYS_ENSEMBLE], dtype=float)
    if arr.sum()==0 or np.all(~np.isfinite(arr)): w = np.ones_like(arr) / len(arr)
    else:
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        w = np.exp(arr / (arr.std() + 1e-6)); w = w / w.sum()
    return {h: float(wi) for h, wi in zip(FUTURE_DAYS_ENSEMBLE, w)}

def build_adaptive_ensemble_safe(risks_by_h: Dict[int, pd.Series], y: pd.Series, lookback:int)\
        ->Tuple[pd.Series, Dict[int,float], Dict[int,float]]:
    base_idx = None
    for r in risks_by_h.values(): base_idx = r.index if base_idx is None else base_idx.intersection(r.index)
    base_idx = base_idx.sort_values()
    w_hist = {h: [] for h in FUTURE_DAYS_ENSEMBLE}; s_hist = {h: [] for h in FUTURE_DAYS_ENSEMBLE}; ens_vals = []
    target_by_h = {H: y.pct_change(H).shift(-H) for H in FUTURE_DAYS_ENSEMBLE}
    for i in range(len(base_idx)):
        t = base_idx[i]; ic_map, sign_map = {}, {}
        for H in FUTURE_DAYS_ENSEMBLE:
            j_end = i - H
            if j_end <= 5: ic_map[H] = 0.0; sign_map[H] = 1.0; continue
            j_start = max(0, j_end - lookback); win = base_idx[j_start:j_end]
            r_win = risks_by_h[H].reindex(win); y_win = target_by_h[H].reindex(win)
            ic, _ = safe_spearman(r_win, y_win)
            if ic != ic: ic = 0.0
            ic_map[H] = float(ic); sign_map[H] = -1.0 if ic > 0 else 1.0
        w_map = _softmax_abs_ic(ic_map); x = 0.0
        for H in FUTURE_DAYS_ENSEMBLE:
            r_t = risks_by_h[H].get(t, np.nan)
            if np.isnan(r_t): continue
            x += sign_map[H] * r_t * w_map[H]
            w_hist[H].append(w_map[H]); s_hist[H].append(sign_map[H])
        ens_vals.append((t, x))
    risk_ens = pd.Series({t: v for t, v in ens_vals}).dropna()
    w_last = {h: (w_hist[h][-1] if w_hist[h] else np.nan) for h in FUTURE_DAYS_ENSEMBLE}
    s_last = {h: (s_hist[h][-1] if s_hist[h] else np.nan) for h in FUTURE_DAYS_ENSEMBLE}
    logger.info("Adaptive weights(last): %s", {h: round(w_last[h],4) if w_last[h]==w_last[h] else None for h in FUTURE_DAYS_ENSEMBLE})
    logger.info("Adaptive signs(last):   %s", s_last)
    return risk_ens, w_last, s_last

# ─────────────────── 8. 메인 ─────────────────── #
def main():
    logger.info("=== KOSPI Risk Index v8.8-fix (features=%s, Dynamic-Directions) ===", SCALING_METHOD)

    # 데이터 로드
    prices = get_price_data(); yields = get_yields(); per_pbr = get_per_pbr()
    flows = get_flow_norm(); kospi = prices["KOSPI"].dropna()

    # [FIXED v8.8-fix] 공통 인덱스 정렬 (kospi가 Series를 유지하도록 수정)
    all_dataframes = [kospi.to_frame('KOSPI'), prices, yields, per_pbr, flows]
    common_index = all_dataframes[0].index
    for df in all_dataframes[1:]:
        common_index = common_index.intersection(df.index)
    
    # 데이터 시작일이 제각각일 수 있으므로, 모든 데이터가 존재하는 가장 늦은 시작일 찾기
    common_start = max(_date_range(c)[0] for c in all_dataframes if _date_range(c)[0] is not None)
    common_index = common_index[common_index >= common_start]
    
    logger.info(f"Common date range: {common_index.min().strftime('%Y-%m-%d')} to {common_index.max().strftime('%Y-%m-%d')}")

    kospi = kospi.reindex(common_index).ffill() # Series 유지
    prices = prices.reindex(common_index).ffill()
    yields = yields.reindex(common_index).ffill()
    per_pbr = per_pbr.reindex(common_index).ffill()
    flows = flows.reindex(common_index).ffill()


    # 튜닝/OOS 구간
    TUNE_END = pd.Timestamp("2018-12-31")
    OOS1_END = pd.Timestamp("2022-12-31")
    idx_tune = kospi.index[kospi.index<=TUNE_END]
    idx_oos1 = kospi.index[(kospi.index>TUNE_END) & (kospi.index<=OOS1_END)]
    idx_oos2 = kospi.index[kospi.index>OOS1_END]

    # [NEW v8.8] 동적 방향성 결정
    # 튜닝 데이터로 방향성 결정을 위한 임시 피처 생성 (가장 일반적인 roll_window 사용)
    temp_features_for_direction = _calculate_features(
        kospi.loc[idx_tune], prices.loc[idx_tune], yields.loc[idx_tune], per_pbr.loc[idx_tune],
        flows.loc[idx_tune], roll_window=252, features_to_flip=[] # 방향성 미적용
    )
    # 대표 호라이즌(126일) 기준으로 방향성 결정
    features_to_flip = get_dynamic_directions(temp_features_for_direction, kospi.loc[idx_tune], horizon=126)

    # Optuna 튜닝
    logger.info("Tuning up to %s with dynamic directions...", TUNE_END.strftime("%Y-%m-%d"))
    sampler = TPESampler(seed=SEED)
    study = optuna.create_study(direction="minimize", pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=3), sampler=sampler)
    features_tune_data = (prices.loc[idx_tune], yields.loc[idx_tune], per_pbr.loc[idx_tune], flows.loc[idx_tune])
    study.optimize(lambda t: _objective(t, kospi.loc[idx_tune], features_tune_data, kospi.loc[idx_tune], features_to_flip),
                   n_trials=OPTUNA_TRIALS)
    best = study.best_params
    logger.info("Optimal params: %s", best)

    # 최적 파라미터로 전체 기간 피처 및 리스크 계산
    roll_window, corr_window, step, model_type, alpha, smooth_lambda = \
        best["roll_window"], best["correlation_window"], best["step"], best["model_type"], best["alpha"], best["smooth_lambda"]
    features_all = _calculate_features(kospi, prices, yields, per_pbr, flows, roll_window, features_to_flip)

    risks_by_h = {}
    for H in FUTURE_DAYS_ENSEMBLE:
        target_all = kospi.pct_change(H).shift(-H)
        w_all = _train_weights_over_time(
            features_all, target_all, corr_window, step, model_type, alpha, smooth_lambda, embargo=H
        )
        risks_by_h[H] = _compute_risk_series(features_all, w_all)

    # 앙상블
    if not ADAPTIVE_ENSEMBLE:
        # 정적 앙상블 로직 (v8.6과 동일, 필요 시 사용)
        pass
    else:
        risks_aligned = {}
        for H, r in risks_by_h.items():
            icb,_ = safe_spearman(r.loc[idx_tune], kospi.loc[idx_tune].pct_change(H).shift(-H))
            risks_aligned[H] = -r if (isinstance(icb,float) and not np.isnan(icb) and icb>0) else r
        risk_ens, w_map, s_map = build_adaptive_ensemble_safe(risks_aligned, kospi, ADAPTIVE_LOOKBACK)
        logger.info("Adaptive ensemble(roll IC, embargo-safe) — weights last: %s", {h: round(w_map[h],4) if w_map.get(h) is not None and w_map[h]==w_map[h] else None for h in FUTURE_DAYS_ENSEMBLE})

    # 평가
    for label, idx_eval in [("OOS1(2019-2022)", idx_oos1), ("OOS2(2023-현재)", idx_oos2), ("ALL", features_all.index)]:
        if len(idx_eval)==0: continue
        _ = ic_all_scales(risk_ens.reindex(idx_eval).dropna(), kospi.loc[idx_eval], best["future_days"], label=label)
        _ = quantile_report(risk_ens.reindex(idx_eval).dropna(), kospi.loc[idx_eval], best["future_days"], q=5, label=label)

    ic_decay_df = ic_decay_curve(risk_ens, kospi, horizons=[21,63,126,252])
    mean_ic, (lo, hi) = block_bootstrap_ic(risk_ens, kospi, horizon=best["future_days"], block=BLOCK_SIZE, n=1000, seed=SEED)
    logger.info("[Bootstrap IC @H=%d] mean=%.4f, 95%% CI=(%.4f,%.4f)", best["future_days"], mean_ic, lo, hi)

    # 저장 및 시각화
    if FINAL_RISK_SCALER == "minmax": risk_scaled = rolling_minmax(risk_ens, 252*5).clip(0, 100)
    elif FINAL_RISK_SCALER == "percentile": risk_scaled = rolling_percentile(risk_ens, 252*5).clip(0, 100)
    else: risk_scaled = risk_ens

    risk_scaled.to_csv("kospi_risk_index_v8.csv")
    pd.Series({f"H{h}_weight": w for h,w in w_map.items()}).to_csv("ensemble_weights_timevarying_last_v8.csv")
    ic_decay_df.to_csv("ic_decay_v8.csv", index=False)
    pd.DataFrame([{"horizon": best["future_days"], "mean": mean_ic, "ci_low": lo, "ci_high": hi}]).to_csv("bootstrap_ic_v8.csv", index=False)
    logger.info("CSV saved: kospi_risk_index_v8.csv, ensemble_weights_timevarying_last_v8.csv, diagnostics CSVs.")

    fig = go.Figure() # 시각화 로직은 이전과 동일
    fig.add_trace(go.Scatter(x=risk_scaled.index, y=risk_scaled, name="Risk Index", line=dict(color="red")))
    fig.add_trace(go.Scatter(x=kospi.index, y=kospi, yaxis="y2", name="KOSPI", line=dict(color="blue", dash="dot"), opacity=0.7))
    fig.update_layout(title=f"KOSPI Risk Index (v8.8-fix, Dynamic Directions)", template="plotly_white", height=700,
                      yaxis=dict(title="Risk (0-100)", range=[0, 100] if FINAL_RISK_SCALER in ("minmax","percentile") else None),
                      yaxis2=dict(title="KOSPI", overlaying="y", side="right"), legend=dict(x=0.01, y=0.99))
    fig.show()

    if not risk_scaled.empty:
        cur = risk_scaled.iloc[-1]
        label = "N/A"
        if FINAL_RISK_SCALER in ("minmax","percentile"):
            label = next(lbl for t, lbl in [(80, "Very High"), (60, "High"), (40, "Neutral"), (20, "Low"), (0, "Very Low")] if cur >= t)
        logger.info("[%s] Current Risk: %.2f (%s)", risk_scaled.index[-1].strftime("%Y-%m-%d"), cur, label)

    logger.info("=== Completed v8.8-fix ===")

if __name__ == "__main__":
    main()
