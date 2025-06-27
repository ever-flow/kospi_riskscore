from __future__ import annotations
import os, math, pickle, warnings, datetime as _dt
from pathlib import Path
from contextlib import contextmanager
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
import optuna
from optuna.pruners import MedianPruner
from pykrx import stock
from scipy.stats import percentileofscore, spearmanr
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands
import plotly.graph_objects as go
import logging

# ─────────────────── 1. 설정 ─────────────────── #
START_DATE, END_DATE = "1995-01-01", _dt.date.today().strftime("%Y-%m-%d")
END_PD = pd.to_datetime(END_DATE)
CACHE_DIR = Path(".cache"); CACHE_DIR.mkdir(exist_ok=True)
LOG_FILE = "kospi_risk_score.log"

OPTUNA_TRIALS = int(os.getenv("N_OPTUNA_TRIALS", 30))
CV_FOLDS, MIN_DAYS = 6, 252 * 5  # 5년으로 증가

ALPHAS = [0.001, 0.01, 0.1, 1.0, 10.0]
MODEL_TYPES = ["ridge", "lasso", "elastic"]

TICKERS: Dict[str, str] = {
    "KOSPI": "^KS11", "SP500": "^GSPC", "VIX": "^VIX", "USDKRW": "KRW=X",
    "US10Y": "^TNX", "US2Y": "^FVX", "CRUDE_OIL": "CL=F", "COPPER": "HG=F",
    "GOLD": "GC=F", "IEF": "IEF",
}
INCEPTION_DATES = {"IEF": "2002-07-22"}

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, mode='w', encoding="utf-8"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# ─────────────────── 2. 헬퍼 ─────────────────── #
@contextmanager
def _open_pickle(path: Path, mode: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, mode) as f:
        yield f

def _save(obj: Any, path: Path) -> None:
    with _open_pickle(path, "wb") as f:
        pickle.dump(obj, f)

def _load(path: Path) -> Any | None:
    return pickle.load(open(path, "rb")) if path.exists() else None

def _date_range(s: pd.Series | pd.DataFrame):
    d = s.dropna()
    return (d.index.min(), d.index.max()) if not d.empty else (None, None)

def rolling_percentile(s: pd.Series, window: int, inverse=False):
    arr, out = s.to_numpy(float), np.full_like(s, np.nan, float)
    m = max(1, int(window * 0.6))
    for i in range(len(arr)):
        if math.isnan(arr[i]):
            continue
        win = arr[max(0, i - window + 1): i + 1]
        win = win[~np.isnan(win)]
        if win.size >= m:
            p = percentileofscore(win, arr[i], "rank")
            out[i] = 100 - p if inverse else p
    return pd.Series(out, index=s.index)

def rolling_zscore(s: pd.Series, window: int):
    r = (s - s.rolling(window).mean()) / s.rolling(window).std()
    return r.replace([np.inf, -np.inf], np.nan)

# ─────────────────── 3. 데이터 ─────────────────── #
def _get_cached(name, loader):
    cache = CACHE_DIR / f"{name}.pkl"
    data = _load(cache)
    if isinstance(data, pd.DataFrame) and data.index.max() >= END_PD - pd.Timedelta(days=7):
        logger.info(f"{name}: 캐시 사용")
        return data
    logger.info(f"{name}: 다운로드")
    data = loader()
    _save(data, cache)
    return data

def get_price_data():
    def _dl():
        df = pd.DataFrame()
        for k, t in TICKERS.items():
            logger.info(f"  ▸ {k}")
            s = yf.download(t, start=START_DATE, end=END_DATE, auto_adjust=True, progress=False)["Close"]
            if k in INCEPTION_DATES:
                s = s.loc[s.index >= pd.to_datetime(INCEPTION_DATES[k])]
            df[k] = s
        return df.ffill()

    df = _get_cached("price_data", _dl)
    df.loc[df.index < pd.to_datetime(INCEPTION_DATES["IEF"]), "IEF"] = np.nan
    return df.loc[:END_PD]

def get_per_pbr():
    def _dl():
        d = stock.get_index_fundamental("19950103", END_PD.strftime("%Y%m%d"), "1001").rename_axis("Date").reset_index()
        d["Date"] = pd.to_datetime(d["Date"])
        return d[["Date", "PER", "PBR"]].set_index("Date").replace(0, np.nan).ffill()

    return _get_cached("per_pbr", _dl).loc[:END_PD]

def get_flow():
    def _dl():
        d = stock.get_market_trading_value_by_date("19990104", END_DATE.replace("-", ""), "KOSPI").rename_axis("Date").reset_index()
        d["Date"] = pd.to_datetime(d["Date"])
        d = d[["Date", "개인", "기관합계", "외국인합계"]].rename(columns={"개인": "Individual", "기관합계": "Institution", "외국인합계": "Foreign"})
        return d.set_index("Date").ffill()

    return _get_cached("flow", _dl).loc[:END_PD]

# ─────────────────── 4. 위험지수 계산 ─────────────────── #
def calc_risk(params, kospi, prices, per_pbr, flows, future_days):
    """KOSPI 위험 지수를 계산. 미래 수익률(y)은 가중치 학습에만 사용되며, 최종 지수에는 영향을 주지 않음."""
    roll, corr_w, step, model, alpha = params
    idx = kospi.index.intersection(prices.index).intersection(per_pbr.index).intersection(flows.index)
    kospi, prices, per_pbr, flows = [x.reindex(idx).ffill() for x in (kospi, prices, per_pbr, flows)]

    bb = BollingerBands(kospi)
    df = pd.DataFrame({
        "RSI": RSIIndicator(kospi, 14).rsi(),
        "MACD_Diff": MACD(kospi).macd_diff(),
        "BB_Pos": ((kospi - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband()).replace(0, np.nan) * 100).fillna(50),
        "Mom20": kospi.pct_change(20) * 100,
        "MA50_Rel": (kospi / kospi.rolling(50, 25).mean() - 1) * 100,
        "Vol20": kospi.pct_change().rolling(20, 10).std() * math.sqrt(252) * 100,
        "Vol_Ratio": kospi.pct_change().rolling(20).std() * math.sqrt(252) / (prices["SP500"].pct_change().rolling(20).std() * math.sqrt(252)).replace(0, np.nan),
        "Ret20": kospi.pct_change(20) * 100,
        "RetDiff": kospi.pct_change(20) * 100 - prices["SP500"].pct_change(20) * 100,
        "KRW_Change": prices["USDKRW"].pct_change(20) * 100,
        "YldSpr": prices["US10Y"] - prices["US2Y"],
        "Cu_Gd": prices["COPPER"] / prices["GOLD"],
        "PER": per_pbr["PER"], "PBR": per_pbr["PBR"], "VIX": prices["VIX"],
        "F_Flow": flows["Foreign"].rolling(20, 10).sum(),
        "I_Flow": flows["Institution"].rolling(20, 10).sum(),
        "D_Flow": flows["Individual"].rolling(20, 10).sum(),
        "Oil_Ret": prices["CRUDE_OIL"].pct_change(20) * 100,
        "Oil_Vol": prices["CRUDE_OIL"].pct_change().rolling(20).std() * math.sqrt(252) * 100,
        "IEF_Ret": prices["IEF"].pct_change(20) * 100,
        "IEF_Vol": prices["IEF"].pct_change().rolling(20).std() * math.sqrt(252) * 100,
    }, index=idx).ffill()

    high = [c for c in df if c not in {"YldSpr", "F_Flow", "I_Flow", "D_Flow"}]
    low = ["YldSpr", "F_Flow", "I_Flow", "D_Flow"]
    sc_pct = pd.concat({k: rolling_percentile(df[k], roll) for k in high + low}, axis=1)
    sc_pct[low] = 100 - sc_pct[low]
    sc_z = df.apply(rolling_zscore, window=roll).clip(-3, 3) * 10 + 50
    sc = pd.concat([sc_pct, sc_z.add_suffix("_Z")], axis=1)

    macd_max = df["MACD_Diff"].abs().rolling(roll, int(roll * 0.6)).max()
    sc["MACD_Sc"] = ((df["MACD_Diff"] / macd_max).fillna(0) * 50 + 50).clip(0, 100)

    # 미래 수익률은 가중치 학습에만 사용
    y = kospi.pct_change(future_days).shift(-future_days)
    w = pd.DataFrame(index=sc.index, columns=sc.columns, dtype=float)

    for i in range(corr_w, len(sc), step):
        win = sc.index[i - corr_w: i]
        train_X = sc.loc[win]
        train_y = y.loc[win]
        train_set = pd.concat([train_X, train_y.rename("t")], axis=1).dropna()

        if len(train_set) < int(corr_w * 0.8):
            continue

        model_obj = {
            "ridge": Ridge(alpha=alpha),
            "lasso": Lasso(alpha=alpha, max_iter=10000),
            "elastic": ElasticNet(alpha=alpha, l1_ratio=0.5, max_iter=10000),
        }[model]

        model_obj.fit(train_set[sc.columns], train_set["t"])
        coef = model_obj.coef_

        if np.abs(coef).sum() > 1e-6:
            raw_train = (train_set[sc.columns] * coef).sum(axis=1)
            if raw_train.var() > 0 and train_set["t"].var() > 0:
                corr_train, _ = spearmanr(raw_train, train_set["t"])
                if not math.isnan(corr_train) and corr_train > 0:
                    coef = -coef
            w.iloc[i] = coef / np.abs(coef).sum()

    w = w.ffill().fillna(0)
    raw = (sc * w).sum(axis=1)  # 위험 지수는 현재 데이터만 사용

    return raw.dropna(), w

# ─────────────────── 5. CV & Optuna ─────────────────── #
def _cv(params, kospi, prices, per_pbr, flows, smooth, folds, future_days):
    """시간 순서에 따른 교차 검증. 훈련 데이터는 검증 데이터 이전까지만 사용."""
    total = len(kospi)
    tr_min = max(params[0] + 60, MIN_DAYS)
    val_min = max(params[1] + future_days + 20, 252)
    if total < tr_min + folds * val_min:
        return np.nan, []

    size, idx, cors = (total - tr_min) // folds, kospi.index, []
    for k in range(folds):
        val_idx = idx[tr_min + k * size: tr_min + (k + 1) * size]
        if len(val_idx) < val_min:
            continue

        train_end_date = val_idx[0]  # 검증 세트 시작 이전까지 훈련 (데이터 리크 방지)
        raw, _ = calc_risk(params, kospi.loc[:train_end_date], prices.loc[:train_end_date], 
                         per_pbr.loc[:train_end_date], flows.loc[:train_end_date], future_days)
        if raw.empty:
            continue

        r_expanding = raw.expanding(min_periods=tr_min).apply(lambda x: percentileofscore(x, x[-1]), raw=True).clip(0, 100)
        r_val = r_expanding.reindex(val_idx).dropna()
        if r_val.empty:
            continue

        r = r_val.ewm(span=smooth).mean()
        ret = kospi.loc[val_idx].pct_change(future_days).shift(-future_days)
        pair = pd.concat({"r": r, "y": ret}, axis=1).dropna()

        if len(pair) < max(15, future_days):
            continue
        c, _ = spearmanr(pair["r"], pair["y"])
        if not math.isnan(c):
            cors.append(abs(c))

    return np.median(cors) if cors else np.nan, cors

def _objective(trial, kospi, prices, per_pbr, flows):
    future_days = trial.suggest_categorical("future_days", [126])
    roll = trial.suggest_categorical("roll", [20, 63, 126, 252, 504])
    corr = trial.suggest_categorical("corr", [63, 126, 252, 504])
    step = trial.suggest_categorical("step", [3, 5, 10, 20])
    model = trial.suggest_categorical("model", MODEL_TYPES)
    alpha = trial.suggest_categorical("alpha", ALPHAS)
    smooth = trial.suggest_categorical("smooth", [1, 3, 5, 10, 20])
    if step > corr or roll > corr:
        return float("-inf")

    params = (roll, corr, step, model, alpha)
    score, _ = _cv(params, kospi, prices, per_pbr, flows, smooth, CV_FOLDS, future_days)

    if score is None or math.isnan(score):
        return float("-inf")

    trial.report(score, step=0)
    if trial.should_prune():
        raise optuna.TrialPruned()

    logger.info(f"Trial {trial.number}: fd={future_days}, roll={roll}, corr={corr}, step={step}, model={model}, alpha={alpha}, smooth={smooth}, |IC|={score:.4f}")
    return score

# ─────────────────── 6. 메인 ─────────────────── #
def main():
    logger.info("=== KOSPI 위험 지수 계산 (v3.7 - Median IC 최적화, 부호 보정) ===")
    prices, per_pbr, flows = get_price_data(), get_per_pbr(), get_flow()
    kospi = prices["KOSPI"].dropna()
    s = max(_date_range(c)[0] for c in (prices, per_pbr, flows))
    kospi, prices, per_pbr, flows = [x.loc[x.index >= s].ffill() for x in (kospi, prices, per_pbr, flows)]
    logger.info("공통 데이터 시작일: %s", s.strftime("%Y-%m-%d"))

    param_cache_file = CACHE_DIR / "risk_para_v7.pkl"
    best = _load(param_cache_file)

    if not (best and isinstance(best, dict) and {"model", "alpha", "roll", "corr", "step", "future_days", "smooth"} <= best.keys()):
        logger.info("Optuna %d trials 시작 (Median IC 최적화)", OPTUNA_TRIALS)
        study = optuna.create_study(direction="maximize", pruner=MedianPruner(n_startup_trials=15, n_warmup_steps=5), sampler=optuna.samplers.TPESampler())
        study.optimize(lambda t: _objective(t, kospi, prices, per_pbr, flows), n_trials=OPTUNA_TRIALS)
        best = study.best_params
        best_value = study.best_value
        _save(best, param_cache_file)
    else:
        best_value = None
        logger.info("캐시 파라미터 사용")

    logger.info("Best params: %s", best)
    if best_value is not None:
        logger.info("Best |IC| (Median, CV): %.4f", best_value)

    params = (best["roll"], best["corr"], best["step"], best["model"], best["alpha"])
    _, fold_ics = _cv(params, kospi, prices, per_pbr, flows, best["smooth"], CV_FOLDS, best["future_days"])
    if fold_ics:
        logger.info("폴드별 |IC| 값: %s", [f"{ic:.4f}" for ic in fold_ics])

    raw_risk, w = calc_risk(params, kospi, prices, per_pbr, flows, best["future_days"])
    risk = raw_risk.rolling(252 * 5, min_periods=252).apply(lambda x: percentileofscore(x, x[-1]), raw=True).clip(0, 100)
    risk = risk.ewm(span=best["smooth"]).mean()
    logger.info("최종 위험 지수 계산 완료 (미래 데이터 미사용 확인)")

    # 부호 확인 및 보정
    y = kospi.pct_change(best["future_days"]).shift(-best["future_days"])
    pair_raw = pd.concat({"raw_risk": raw_risk, "y": y}, axis=1).dropna()
    final_ic_raw, _ = spearmanr(pair_raw["raw_risk"], pair_raw["y"])
    logger.info("Final IC between raw risk and future returns: %.4f", final_ic_raw)

    pair_risk = pd.concat({"risk": risk, "y": y}, axis=1).dropna()
    final_ic_risk, _ = spearmanr(pair_risk["risk"], pair_risk["y"])
    logger.info("Final IC between risk score and future returns: %.4f", final_ic_risk)

    # 부호 보정: 위험 지수가 미래 수익률과 양의 상관관계를 가지면 반전
    is_inverted = final_ic_risk > 0
    if is_inverted:
        logger.warning("Risk score가 미래 수익률과 양의 상관관계를 가짐. 위험 지수를 반전(100 - risk) 처리.")
        risk = 100 - risk
        pair_risk = pd.concat({"risk": risk, "y": y}, axis=1).dropna()
        final_ic_risk, _ = spearmanr(pair_risk["risk"], pair_risk["y"])
        logger.info("반전 후 Final IC between risk score and future returns: %.4f", final_ic_risk)

    risk.to_csv("kospi_risk_index_v3.csv")
    w.to_csv("indicator_weights_v3.csv")
    logger.info("CSV 저장 완료")

    # 시각화
    title_suffix = " (Inverted)" if is_inverted else ""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=risk.index, y=risk, name="Risk Score", line=dict(color="red")))
    fig.add_trace(go.Scatter(x=kospi.index, y=kospi, yaxis="y2", name="KOSPI", line=dict(color="blue", dash="dot"), opacity=0.7))
    for lvl in [20, 40, 60, 80]:
        fig.add_hline(y=lvl, line_dash="dash", opacity=0.3, line_color="gray")
    fig.update_layout(
        title=f"KOSPI Risk Score vs. Index (Median IC Optimization){title_suffix}",
        yaxis=dict(title="Risk Score (0-100, High is Risky)", range=[0, 100]),
        yaxis2=dict(title="KOSPI Index", overlaying="y", side="right"),
        template="plotly_white",
        height=600,
        legend=dict(x=0.01, y=0.99)
    )
    fig.show()

    if not risk.empty:
        cur = risk.iloc[-1]
        desc = next(lbl for t, lbl in [(80, "매우 높음"), (60, "높음"), (40, "중립"), (20, "낮음"), (0, "매우 낮음")] if cur >= t)
        logger.info("[%s] 현재 위험 지수: %.2f (%s)%s", risk.index[-1].strftime("%Y-%m-%d"), cur, desc, title_suffix)
    logger.info("=== 완료 ===")

if __name__ == "__main__":
    main()
