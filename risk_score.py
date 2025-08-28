
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
ALPHA_MIN, ALPHA_MAX= 0.1, 50.0 # 규제 강화
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
FINAL_RISK_SCALER = "minmax"  # ("minmax","percentile","none")

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

# [FIX-A] 미래정보 누수 방지용 롤링+시프트 winsorize
def rolling_winsorize_shift(s: pd.Series, window:int=252, p:float=0.01) -> pd.Series:
    """
    현재 시점 이전 데이터(shift(1))의 롤링 분위수로 컷오프 산정 → look-ahead 방지.
    """
    lo = s.shift(1).rolling(window, min_periods=max(20, window//2)).quantile(p)
    hi = s.shift(1).rolling(window, min_periods=max(20, window//2)).quantile(1-p)
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

    # [FIX-A] 전구간 winsorize → 롤링+시프트 winsorize로 교체 (look-ahead 방지)
    out = out.apply(lambda s: rolling_winsorize_shift(s, window=252, p=0.01)).clip(0, 1)

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

# ─────────────────── 3. 특징 계산 ─────────────────── #
_FEATURES_CACHE: Dict[str, pd.DataFrame] = {}

def _features_cache_key(kospi_idx: pd.Index, roll_window:int, flips:List[str])->str:
    h = hashlib.md5((",".join(str(x) for x in kospi_idx[:10]) + "|" + str(len(kospi_idx))).encode()).hexdigest()
    flip_key = ",".join(sorted(flips)) if flips else "-"
    return f"{h}|roll={roll_window}|scale={SCALING_METHOD}|flip={flip_key}"

def get_dynamic_directions(features_df: pd.DataFrame, y: pd.Series, horizon: int) -> List[str]:
    """
    동적 방향성: 미래 수익률(H)과 양(+) 상관이면 '리스크'로 보기 위해 뒤집는다.
    [FIX-B 유지] RegimeBull은 flip 대상에서 제외한다.
    """
    target = y.pct_change(horizon).shift(-horizon)
    positive_corr_features = []
    logger.info(f"Calculating dynamic directions over {y.index.min().year}-{y.index.max().year} for H={horizon}...")
    for col in features_df.columns:
        if col == "RegimeBull":  # exclude
            continue
        ic, _ = safe_spearman(features_df[col], target)
        if isinstance(ic, float) and not np.isnan(ic) and ic > 0:
            positive_corr_features.append(col)
            logger.info(f"  ▸ Feature '{col}' has positive correlation ({ic:.3f}). Will be inverted.")
    return positive_corr_features

def _calculate_features(
    kospi: pd.Series, prices: pd.DataFrame, yields: pd.DataFrame, per_pbr: pd.DataFrame,
    flows_norm: pd.DataFrame, roll_window: int, features_to_flip: List[str]
)->pd.DataFrame:
    idx = kospi.index
    for x in (prices, yields, per_pbr, flows_norm):
        idx = idx.intersection(x.index)
    kospi, prices, yields, per_pbr, flows = [x.reindex(idx).ffill() for x in (kospi, prices, yields, per_pbr, flows_norm)]

    key = _features_cache_key(kospi.index, roll_window, features_to_flip)
    if key in _FEATURES_CACHE:
        return _FEATURES_CACHE[key].copy()

    yld_spr = (yields["US10Y"] - yields["US2Y"]).rename("YldSpr")

    df = pd.DataFrame({
        "Mom20":      kospi.pct_change(20),
        "MA50_Rel":   kospi / kospi.rolling(50, min_periods=50).mean() - 1.0,
        "Vol20":      kospi.pct_change().rolling(20, min_periods=20).std() * np.sqrt(252),
        "VolRatio":   (kospi.pct_change().rolling(20, min_periods=20).std() * np.sqrt(252)) /
                      (prices["SP500"].pct_change().rolling(20, min_periods=20).std() * np.sqrt(252)).replace(0, np.nan),
        "Ret20":      kospi.pct_change(20),
        "RetDiff":    kospi.pct_change(20) - prices["SP500"].pct_change(20),
        "KRW_Change": prices["USDKRW"].pct_change(20),
        "YldSpr":     yld_spr,
        "Cu_Gd":      (prices["COPPER"] / prices["GOLD"]).replace([np.inf, -np.inf], np.nan),
        "PER":        per_pbr["PER"], "PBR": per_pbr["PBR"], "VIX": prices["VIX"],
        "F_FlowR":    flows["F_FlowR"], "I_FlowR": flows["I_FlowR"], "D_FlowR": flows["D_FlowR"],
        "Oil_Ret":    prices["CRUDE_OIL"].pct_change(20), "IEF_Ret": prices["IEF"].pct_change(20),
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

    # RegimeBull 스케일 정합(0/100) 후 피처에 추가
    sma200 = kospi.rolling(200, min_periods=200).mean()
    regime = (kospi >= sma200).astype(float)
    if SCALING_METHOD in ("percentile", "z"):
        regime = (regime * 100.0).rename("RegimeBull")   # 0/100로 정렬
    else:
        regime = regime.rename("RegimeBull")
    scaled = pd.concat([scaled, regime.reindex(scaled.index)], axis=1)

    # Step 2: Apply dynamic direction flip (RegimeBull 제외)
    for col in features_to_flip:
        if col == "RegimeBull":
            continue
        if col in scaled.columns:
            if SCALING_METHOD in ("percentile", "z"):
                scaled[col] = 100 - scaled[col]
            else:
                scaled[col] = -scaled[col]

    scaled = scaled.dropna(how='all')
    _FEATURES_CACHE[key] = scaled.copy()
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
    """
    반환: 날짜×피처의 가중치(절댓값 합=1로 정규화, ffill)
    """
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
    # We MINIMIZE IC because more negative IC = better (risk↑ → return↓)
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
def _softmax_abs_ic(ics: Dict[int,float], temp: float = 0.5, floor: float = 0.1)->Dict[int,float]:
    """
    [High fix] Use fixed temperature + z-score + optional floor to avoid weight collapse to a single H.
    """
    x = np.array([abs(ics.get(h,0.0)) for h in FUTURE_DAYS_ENSEMBLE], dtype=float)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if (not np.isfinite(x).any()) or np.allclose(x.sum(), 0.0):
        w = np.ones_like(x) / len(x)
    else:
        x = (x - x.mean()) / (x.std() + 1e-6)       # z-score
        w = np.exp(x / max(1e-6, temp))
        w = w / w.sum()
        if floor and floor > 0:
            w = np.clip(w, floor, None)
            w = w / w.sum()
    return {h: float(wi) for h, wi in zip(FUTURE_DAYS_ENSEMBLE, w)}

def build_adaptive_ensemble_safe(risks_by_h: Dict[int, pd.Series], y: pd.Series, lookback:int)\
        ->Tuple[pd.Series, Dict[int,float], Dict[int,float], pd.DataFrame, pd.DataFrame]:
    """
    반환:
      - risk_ens: 적응형 앙상블 리스크
      - w_last, s_last: 마지막 시점의 H별 가중치/부호
      - w_hist_df, s_hist_df: 전기간의 H별 가중치/부호 (시각화용)
    """
    base_idx = None
    for r in risks_by_h.values(): base_idx = r.index if base_idx is None else base_idx.intersection(r.index)
    base_idx = base_idx.sort_values()
    w_hist = {h: [] for h in FUTURE_DAYS_ENSEMBLE}; s_hist = {h: [] for h in FUTURE_DAYS_ENSEMBLE}; ens_vals = []
    target_by_h = {H: y.pct_change(H).shift(-H) for H in FUTURE_DAYS_ENSEMBLE}
    for i in range(len(base_idx)):
        t = base_idx[i]; ic_map, sign_map = {}, {}
        for H in FUTURE_DAYS_ENSEMBLE:
            j_end = i - H
            if j_end <= 5: 
                ic_map[H] = 0.0; sign_map[H] = 1.0
            else:
                j_start = max(0, j_end - lookback); win = base_idx[j_start:j_end]
                r_win = risks_by_h[H].reindex(win); y_win = target_by_h[H].reindex(win)
                ic, _ = safe_spearman(r_win, y_win); ic = 0.0 if np.isnan(ic) else float(ic)
                ic_map[H] = ic; sign_map[H] = -1.0 if ic > 0 else 1.0
        w_map = _softmax_abs_ic(ic_map, temp=0.5, floor=0.1)   # ← 고정 온도/하한
        x = 0.0
        for H in FUTURE_DAYS_ENSEMBLE:
            r_t = risks_by_h[H].get(t, np.nan)
            if not np.isnan(r_t):
                x += sign_map[H] * r_t * w_map[H]
            w_hist[H].append(w_map[H]); s_hist[H].append(sign_map[H])
        ens_vals.append((t, x))
    risk_ens = pd.Series({t: v for t, v in ens_vals}).dropna()
    w_last = {h: (w_hist[h][-1] if w_hist[h] else np.nan) for h in FUTURE_DAYS_ENSEMBLE}
    s_last = {h: (s_hist[h][-1] if s_hist[h] else np.nan) for h in FUTURE_DAYS_ENSEMBLE}
    # 타임시리즈 DataFrame 구성
    w_hist_df = pd.DataFrame({f"H{h}": w_hist[h] for h in FUTURE_DAYS_ENSEMBLE}, index=base_idx)
    s_hist_df = pd.DataFrame({f"H{h}": s_hist[h] for h in FUTURE_DAYS_ENSEMBLE}, index=base_idx)
    logger.info("Adaptive weights(last): %s", {h: round(w_last[h],4) if w_last[h]==w_last[h] else None for h in FUTURE_DAYS_ENSEMBLE})
    logger.info("Adaptive signs(last):   %s", s_last)
    return risk_ens, w_last, s_last, w_hist_df, s_hist_df

# ─────────────────── 8. 메인 ─────────────────── #
def main(topn_features_for_plot:int=8):
    logger.info("=== KOSPI Risk Index v8.8-fix-A&B (Critical+High fixes applied) ===")

    # 데이터 로드
    prices = get_price_data(); yields = get_yields(); per_pbr = get_per_pbr()
    flows = get_flow_norm(); kospi = prices["KOSPI"].dropna()

    # 공통 인덱스 정렬 (kospi가 Series 유지)
    all_dataframes = [kospi.to_frame('KOSPI'), prices, yields, per_pbr, flows]
    common_index = all_dataframes[0].index
    for df in all_dataframes[1:]:
        common_index = common_index.intersection(df.index)
    common_start = max(_date_range(c)[0] for c in all_dataframes if _date_range(c)[0] is not None)
    common_index = common_index[common_index >= common_start]
    logger.info(f"Common date range: {common_index.min().strftime('%Y-%m-%d')} to {common_index.max().strftime('%Y-%m-%d')}")

    kospi = kospi.reindex(common_index).ffill()
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

    # 동적 방향성 결정(대표 H=126, RegimeBull 제외)
    temp_features_for_direction = _calculate_features(
        kospi.loc[idx_tune], prices.loc[idx_tune], yields.loc[idx_tune], per_pbr.loc[idx_tune],
        flows.loc[idx_tune], roll_window=252, features_to_flip=[]
    )
    features_to_flip = get_dynamic_directions(temp_features_for_direction, kospi.loc[idx_tune], horizon=126)

    # Optuna 튜닝 (IC minimize → 더 음수일수록 좋음)
    logger.info("Tuning up to %s with dynamic directions...", TUNE_END.strftime("%Y-%m-%d"))
    sampler = TPESampler(seed=SEED)
    study = optuna.create_study(direction="minimize", pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=3), sampler=sampler)
    features_tune_data = (prices.loc[idx_tune], yields.loc[idx_tune], per_pbr.loc[idx_tune], flows.loc[idx_tune])
    study.optimize(lambda t: _objective(t, kospi.loc[idx_tune], features_tune_data, kospi.loc[idx_tune], features_to_flip),
                   n_trials=OPTUNA_TRIALS)
    best = study.best_params
    logger.info("Optimal params: %s", best)

    # 최적 파라미터로 전체 피처/리스크 계산
    roll_window, corr_window, step, model_type, alpha, smooth_lambda = \
        best["roll_window"], best["correlation_window"], best["step"], best["model_type"], best["alpha"], best["smooth_lambda"]
    features_all = _calculate_features(kospi, prices, yields, per_pbr, flows, roll_window, features_to_flip)

    risks_by_h = {}
    weights_by_h = {}   # 시각화 저장용
    for H in FUTURE_DAYS_ENSEMBLE:
        target_all = kospi.pct_change(H).shift(-H)
        w_all = _train_weights_over_time(
            features_all, target_all, corr_window, step, model_type, alpha, smooth_lambda, embargo=H
        )
        weights_by_h[H] = w_all
        risks_by_h[H] = _compute_risk_series(features_all, w_all)

    # [Critical fix] 부호 정렬 단계 제거 → 앙상블에서만 부호 동적 처리
    if not ADAPTIVE_ENSEMBLE:
        raise NotImplementedError("Static ensemble path is disabled in this build.")
    else:
        risk_ens, w_last, s_last, w_hist_df, s_hist_df = build_adaptive_ensemble_safe(risks_by_h, kospi, ADAPTIVE_LOOKBACK)

    # 평가 (원래 OOS 구간)
    for label, idx_eval in [("OOS1(2019-2022)", idx_oos1), ("OOS2(2023-현재)", idx_oos2), ("ALL", features_all.index)]:
        if len(idx_eval)==0: continue
        _ = ic_all_scales(risk_ens.reindex(idx_eval).dropna(), kospi.loc[idx_eval], best["future_days"], label=label)
        _ = quantile_report(risk_ens.reindex(idx_eval).dropna(), kospi.loc[idx_eval], best["future_days"], q=5, label=label)

    # [Equal-length OOS] OOS1 vs OOS2 동일 길이 비교
    len_oos1, len_oos2 = len(idx_oos1), len(idx_oos2)
    eq_len = min(len_oos1, len_oos2)
    if eq_len > 0:
        idx_oos1_eq = idx_oos1[-eq_len:]
        idx_oos2_eq = idx_oos2[-eq_len:]
        for label, idx_eval in [(f"OOS1_EQ(last {eq_len} days)", idx_oos1_eq),
                                (f"OOS2_EQ(last {eq_len} days)", idx_oos2_eq)]:
            _ = ic_all_scales(risk_ens.reindex(idx_eval).dropna(), kospi.loc[idx_eval], best["future_days"], label=label)
            _ = quantile_report(risk_ens.reindex(idx_eval).dropna(), kospi.loc[idx_eval], best["future_days"], q=5, label=label)

    ic_decay_df = ic_decay_curve(risk_ens, kospi, horizons=[21,63,126,252])
    mean_ic, (lo, hi) = block_bootstrap_ic(risk_ens, kospi, horizon=best["future_days"], block=BLOCK_SIZE, n=1000, seed=SEED)
    logger.info("[Bootstrap IC @H=%d] mean=%.4f, 95%% CI=(%.4f,%.4f)", best["future_days"], mean_ic, lo, hi)

    # 저장
    if FINAL_RISK_SCALER == "minmax": risk_scaled = rolling_minmax(risk_ens, 252*5).clip(0, 100)
    elif FINAL_RISK_SCALER == "percentile": risk_scaled = rolling_percentile(risk_ens, 252*5).clip(0, 100)
    else: risk_scaled = risk_ens

    risk_scaled.to_csv("kospi_risk_index_v8.csv")
    pd.Series({f"H{h}_weight_last": w for h,w in w_last.items()}).to_csv("ensemble_weights_last_v8.csv")
    w_hist_df.to_csv("ensemble_weights_timeseries_v8.csv")
    s_hist_df.to_csv("ensemble_signs_timeseries_v8.csv")
    ic_decay_df.to_csv("ic_decay_v8.csv", index=False)
    pd.DataFrame([{"horizon": best["future_days"], "mean": mean_ic, "ci_low": lo, "ci_high": hi}]).to_csv("bootstrap_ic_v8.csv", index=False)
    logger.info("CSV saved: kospi_risk_index_v8.csv, ensemble_weights_last_v8.csv, ensemble_weights_timeseries_v8.csv, diagnostics CSVs.")

    # ── 시각화 1: 리스크 지수 & KOSPI
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=risk_scaled.index, y=risk_scaled, name="Risk Index", line=dict(color="red")))
    fig.add_trace(go.Scatter(x=kospi.index, y=kospi, yaxis="y2", name="KOSPI", line=dict(color="blue", dash="dot"), opacity=0.7))
    fig.update_layout(title=f"KOSPI Risk Index (v8.8-fix-A&B, Critical+High fixes)",
                      template="plotly_white", height=700,
                      yaxis=dict(title="Risk (0-100)", range=[0, 100] if FINAL_RISK_SCALER in ("minmax","percentile") else None),
                      yaxis2=dict(title="KOSPI", overlaying="y", side="right"), legend=dict(x=0.01, y=0.99))
    fig.show()

    # ── 시각화 2: 적응형 앙상블 H별 가중치 타임시리즈
    fig_w = go.Figure()
    for H in FUTURE_DAYS_ENSEMBLE:
        col = f"H{H}"
        if col in w_hist_df.columns:
            fig_w.add_trace(go.Scatter(x=w_hist_df.index, y=w_hist_df[col], name=f"{col} weight"))
    fig_w.update_layout(title="Adaptive Ensemble Weights over Time (H=63/126/252)", template="plotly_white", height=500,
                        yaxis=dict(title="Weight (softmax on |IC|)"))
    fig_w.show()

    # ── 시각화 3: 지배적 H의 피처 가중치 Top-N 시리즈(라인)
    # 마지막 시점에서 가장 큰 가중치를 가진 H 선택
    dominant_H = max(w_last, key=lambda h: (w_last[h] if w_last[h]==w_last[h] else -1))
    w_dom = weights_by_h[dominant_H]  # 날짜×피처
    # 최근 가중치(마지막 유효일)의 절댓값 기준 Top-N 피처 선택
    last_row = w_dom.dropna(how="all").iloc[-1]
    top_feats = last_row.abs().sort_values(ascending=False).head(topn_features_for_plot).index.tolist()

    fig_fw = go.Figure()
    for c in top_feats:
        fig_fw.add_trace(go.Scatter(x=w_dom.index, y=w_dom[c], name=f"{c} (H{dominant_H})", mode="lines"))
    fig_fw.update_layout(title=f"Top-{topn_features_for_plot} Feature Weights over Time (Dominant H={dominant_H})",
                         template="plotly_white", height=600, yaxis=dict(title="Normalized weight (|w| sum = 1)"))
    fig_fw.show()

    # 현재 리스크 레벨 로그
    if not risk_scaled.empty:
        cur = risk_scaled.iloc[-1]
        label = "N/A"
        if FINAL_RISK_SCALER in ("minmax","percentile"):
            label = next(lbl for t, lbl in [(80, "Very High"), (60, "High"), (40, "Neutral"), (20, "Low"), (0, "Very Low")] if cur >= t)
        logger.info("[%s] Current Risk: %.2f (%s)", risk_scaled.index[-1].strftime("%Y-%m-%d"), cur, label)

    logger.info("=== Completed (Critical+High fixes) ===")

if __name__ == "__main__":
    main()
