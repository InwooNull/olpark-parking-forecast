"""한국체육산업개발 공개 파일데이터 4종 → 앱용 data.json
입력: data_raw/ (data.go.kr 15138735 입출차, 15164156 혼잡도, 15131671 월별주차현황, 15149169 영문 공연정보)
"""
import json
from pathlib import Path
import numpy as np, pandas as pd

R = Path(__file__).resolve().parent.parent / "data_raw"
def rd(pat): return pd.read_csv(next(R.glob(pat)), encoding="cp949")

io = rd("15138735_*.csv")
io["in"] = pd.to_datetime(io["입차일시"]); io["out"] = pd.to_datetime(io["출차일시"])
rng = pd.date_range("2024-07-01", "2024-08-01", freq="15min", inclusive="left")
ins = np.sort(io["in"].values); outs = np.sort(io["out"].values)
occ = pd.Series(np.searchsorted(ins, rng.values, side="right") - np.searchsorted(outs, rng.values, side="right"), index=rng)
arr = io.set_index("in").resample("15min").size().reindex(rng, fill_value=0)
dep = io.set_index("out").resample("15min").size().reindex(rng, fill_value=0)

daypeak = occ.resample("D").max()
EVENT_PEAK = 1500  # 동시 주차 1,500대 이상인 날 = 대형 행사일
event_days = sorted(d.strftime("%Y-%m-%d") for d in daypeak[daypeak >= EVENT_PEAK].index)
def dtype(ts):
    d = ts.strftime("%Y-%m-%d")
    if d in event_days: return "event"
    return "weekend" if ts.weekday() >= 5 else "weekday"
kinds = pd.Series([dtype(t) for t in rng], index=rng)
# 7/1은 6월에 들어온 차가 빠져 있어(데이터가 입차일 기준) 동시 주차가 낮게 잡힘 → 프로파일에서 제외
kinds[kinds.index < "2024-07-02"] = "warmup"
slot = (rng.hour * 4 + rng.minute // 15)
prof = {}
for k in ["weekday", "weekend", "event"]:
    m = kinds.values == k  # warmup(7/1) 제외
    df = pd.DataFrame({"slot": slot[m], "occ": occ.values[m], "arr": arr.values[m], "dep": dep.values[m]})
    g = df.groupby("slot").mean().reindex(range(96), fill_value=0)
    prof[k] = {c: [round(float(v), 1) for v in g[c]] for c in ["occ", "arr", "dep"]}
    prof[k]["days"] = int(len(set(rng[m].date)))
q = None  # 등급은 아래 혼잡도 기록으로 보정

# 월별 계수 (올림픽공원, 3년 평균 / 7월 평균)
m = rd("15131671_*.csv"); m = m[m["주차장소"] == "올림픽공원"].copy(); m["월"] = m["월"].str.replace("월", "").astype(int)
# 프로파일은 2024년 7월 값이므로 분모도 2024-07, 이상치 영향을 줄이려고 3년 중앙값 사용
base = int(m[(m["연도"] == 2024) & (m["월"] == 7)]["주차대수"].iloc[0])
mm = m.groupby("월")["주차대수"].median()
month_factor = {int(k): round(float(v / base), 3) for k, v in mm.items()}
monthly = {int(y): {int(r["월"]): int(r["주차대수"]) for _, r in g.iterrows()} for y, g in m.groupby("연도")}

# 구역별 혼잡도 기록
cg = rd("15164156_*.csv")
cg["h"] = pd.to_datetime(cg["등록시간"], format="%H:%M:%S", errors="coerce").dt.hour
# 검증·등급 보정: 혼잡도 기록 시점의 예측값과 실제 라벨 비교(행사일 패턴은 쓰지 않음)
cg["dt"] = pd.to_datetime(cg["등록일"] + " " + cg["등록시간"])
def pred(t):
    k = "weekend" if t.weekday() >= 5 else "weekday"
    return prof[k]["occ"][t.hour * 4 + t.minute // 15] * month_factor[t.month]
cg["p"] = cg["dt"].apply(pred)
EDGES = [900, 1050, 1200]
lvl = np.digitize(cg["p"], EDGES)
calib = [{"lo": ([0] + EDGES)[i], "hi": (EDGES + [None])[i], "n": int((lvl == i).sum()),
          "full_rate": round(float((cg["혼잡도"][lvl == i] == "만차").mean()), 3),
          "busy_rate": round(float((cg["혼잡도"][lvl == i] != "원활").mean()), 3)} for i in range(4)]
order = cg["혼잡도"].map({"원활": 0, "혼잡": 1, "만차": 2})
validation = {"n": int(len(cg)), "spearman": round(float(cg["p"].corr(order, method="spearman")), 3),
              "median_pred": {k: int(v) for k, v in cg.groupby("혼잡도")["p"].median().items()}}
zones = {}
for z, g in cg.groupby("주차구역"):
    n = len(g); vc = g["혼잡도"].value_counts()
    zones[z] = {"n": n, "full": int(vc.get("만차", 0)), "busy": int(vc.get("혼잡", 0)), "free": int(vc.get("원활", 0)),
                "full_rate": round(vc.get("만차", 0) / n, 3)}
cg_period = [str(cg["등록일"].min()), str(cg["등록일"].max())]

# 공연 일정
pf = rd("15149169_*.csv"); pf["d"] = pd.to_datetime(pf["공연일자"], errors="coerce")
events = sorted({(r["d"].strftime("%Y-%m-%d"), str(r["공연제목"])) for _, r in pf.dropna(subset=["d"]).iterrows()})
ev_by_date = {}
for d, t in events: ev_by_date.setdefault(d, []).append(t)

stay = (io["out"] - io["in"]).dt.total_seconds() / 60
io["kind"] = [("event" if d in event_days else ("weekend" if t.weekday() >= 5 else "weekday")) for d, t in zip(io["in"].dt.strftime("%Y-%m-%d"), io["in"])]
eve = io[(io["in"].dt.hour >= 17) & (io["in"].dt.hour < 21)]
short_stay = {k: round(float(((eve["out"] - eve["in"]).dt.total_seconds() / 60 <= 10)[eve["kind"] == k].mean()), 3) for k in ["weekday", "weekend", "event"]}
out = {
    "source": {
        "inout": "한국체육산업개발(주)_올림픽공원 입출차 정보_20240731 (data.go.kr 15138735), 2024-07-01~31, %d건" % len(io),
        "congestion": "한국체육산업개발(주)_주차장 혼잡도 정보_20260812 (data.go.kr 15164156), %s~%s, %d건" % (cg_period[0], cg_period[1], len(cg)),
        "monthly": "한국체육산업개발(주)_월별주차현황_20251231 (data.go.kr 15131671), 2023~2025",
        "events": "한국체육산업개발(주)_영문 공연정보_20260901 (data.go.kr 15149169), %d건" % len(pf),
    },
    "profiles": prof, "level_edges": EDGES, "calibration": calib, "validation": validation, "short_stay_17_21": short_stay, "month_base_2024_07": base, "event_peak_rule": EVENT_PEAK, "event_days_2024_07": event_days,
    "daypeak_2024_07": {d.strftime("%Y-%m-%d"): int(v) for d, v in daypeak.items()},
    "month_factor": month_factor, "monthly": monthly, "zones": zones, "events": ev_by_date,
    "stats": {"entries": int(len(io)), "daily_mean": round(len(io) / 31), "stay_median_min": round(float(stay.median())),
              "peak_max": int(occ.max()), "peak_max_at": str(occ.idxmax())},
}
p = Path(__file__).resolve().parent.parent / "data.json"
p.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
print("saved", p, p.stat().st_size, "bytes"); print("event days", event_days); print("calib", calib); print("validation", validation); print("short", short_stay)
for k in prof: print(k, prof[k]["days"], "peak", max(prof[k]["occ"]), "at slot", int(np.argmax(prof[k]["occ"])))
print({z: zones[z]["full_rate"] for z in sorted(zones, key=lambda z: -zones[z]["full_rate"])})
print(month_factor)
