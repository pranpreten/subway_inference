# -*- coding: utf-8 -*-
"""
지하철 승하차 수요예측 — 추론 대시보드 (로컬 웹서버)

실행:  python app.py   →  브라우저에서 http://localhost:5000
필요:  pip install flask  (pandas, lightgbm, joblib, holidays는 학습 때 설치됨)

구조:
  - 시작 시 ../models/models_v3.pkl (모델 8개) 로드
  - /api/features : 선택한 일시 → 파생피처 계산 결과 반환 (모델 입력 미리보기)
  - /api/predict  : 같은 일시 → 모델 8개 추론 (오늘부터 미래 아무 시각이나 가능)
  - 피처 계산 로직은 전처리 노트북 셀 7~8과 동일 (일시 하나만으로 전부 계산)
  - 가드: 운행창(05:00~익일 00:10) 밖 시각은 거부, 음수 예측은 단조성 보정에서 0으로 클리핑
"""
from datetime import date, timedelta
from pathlib import Path

import holidays
import joblib
import pandas as pd
from flask import Flask, jsonify, render_template_string, request

# ---------------------------------------------------------------- 준비물 로드
BASE = Path(__file__).resolve().parent.parent          # 지하철관련/
MODELS = joblib.load(BASE / 'models' / 'models_v3.pkl')
FEATURES = list(MODELS['y_board_60'].feature_name_)    # 모델이 기억하는 피처 목록 그대로 사용
TARGETS = list(MODELS.keys())

_kr = holidays.KR(years=range(2024, 2041), language='ko')   # 미래 입력 대비 넉넉히
_hol = set(pd.to_datetime(list(_kr.keys())))
_major = set(pd.to_datetime([d for d, n in _kr.items() if ('설날' in n) or ('추석' in n)]))

# 피처 한글 설명 (화면 표시용)
FEATURE_LABELS = {
    'hour': '시각 (0~23)',
    'slot': '하루 내 5분 칸 번호 (0~287)',
    'dow': '요일 (0=월 ~ 6=일)',
    'month': '월',
    'is_weekend': '주말 여부',
    'is_holiday': '공휴일 여부',
    'is_major_holiday': '명절 연휴 (설·추석)',
    'is_minor_holiday': '일반 공휴일',
    'holiday_prev': '어제가 공휴일',
    'holiday_next': '내일이 공휴일',
    'is_sandwich': '샌드위치데이',
    'is_yearend': '연말 (12/24~31)',
    'trend': '2024-01-01부터 경과일',
}


def _off(day):
    """쉬는 날 여부 (공휴일 또는 주말)"""
    return (day in _hol) or (day.dayofweek >= 5)


def check_window(t: pd.Timestamp):
    """운행창(05:00 ~ 익일 00:10) 검증 — 학습 데이터에 없는 시간대는 예측값이 무의미"""
    minutes = t.hour * 60 + t.minute
    ok = (minutes >= 5 * 60) or (minutes <= 10)   # 05:00~23:59 또는 00:00~00:10
    if not ok:
        return f'{t:%H:%M}는 운행창(05:00~익일 00:10) 밖입니다. 학습 범위 밖이라 예측 불가.'
    return None


def make_features(t: pd.Timestamp) -> dict:
    """전처리 셀 7~8과 동일한 로직 — 일시 하나로 파생피처 전부 계산"""
    day = t.normalize()
    prev, nxt = day - pd.Timedelta('1D'), day + pd.Timedelta('1D')
    feats = {
        'hour': t.hour,
        'slot': t.hour * 12 + t.minute // 5,
        'dow': t.dayofweek,
        'month': t.month,
        'is_weekend': int(t.dayofweek >= 5),
        'is_holiday': int(day in _hol),
        'is_major_holiday': int(day in _major),
        'is_minor_holiday': int((day in _hol) and (day not in _major)),
        'holiday_prev': int(prev in _hol),
        'holiday_next': int(nxt in _hol),
        'is_sandwich': int((not _off(day)) and _off(prev) and _off(nxt)),
        'is_yearend': int(t.month == 12 and t.day >= 24),
        'trend': (day - pd.Timestamp('2024-01-01')).days,
    }
    return {k: feats[k] for k in FEATURES}   # 모델이 아는 피처만, 학습 때 순서로


def predict(t: pd.Timestamp) -> dict:
    """모델 8개 추론 + 단조성 보정(5≤15≤30≤60)"""
    x = pd.DataFrame([make_features(t)])
    # categorical은 학습 때와 같은 전체 범주로 선언 (한 행짜리 입력의 필수 처리)
    x['dow'] = pd.Categorical(x['dow'], categories=range(7))
    x['month'] = pd.Categorical(x['month'], categories=range(1, 13))

    out = {}
    for col in ['board', 'alight']:
        run = 0.0
        for w in [5, 15, 30, 60]:
            run = max(run, float(MODELS[f'y_{col}_{w}'].predict(x)[0]))
            out[f'{col}_{w}'] = round(run)
    return out


# ---------------------------------------------------------------- Flask 앱
app = Flask(__name__)


@app.route('/api/meta')
def api_meta():
    today = date.today()
    return jsonify({
        'date_min': str(today),                              # 오늘부터
        'date_max': str(today + timedelta(days=365 * 3)),    # 3년 뒤까지 (달력 피처라 제한은 UI용)
        'features': FEATURES,
    })


@app.route('/api/features')
def api_features():
    t = pd.Timestamp(request.args['ts'])
    err = check_window(t)
    if err:
        return jsonify({'error': err}), 400
    feats = make_features(t)
    return jsonify({
        'features': [
            {'name': k, 'label': FEATURE_LABELS.get(k, k), 'value': v}
            for k, v in feats.items()
        ],
        'dow_name': '월화수목금토일'[t.dayofweek],
    })


@app.route('/api/predict')
def api_predict():
    t = pd.Timestamp(request.args['ts'])
    err = check_window(t)
    if err:
        return jsonify({'error': err}), 400
    return jsonify({'pred': predict(t)})


@app.route('/')
def index():
    return render_template_string(PAGE)


# ---------------------------------------------------------------- 화면 (단일 페이지)
PAGE = """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<title>지하철 수요예측 추론 대시보드</title>
<style>
  * { box-sizing: border-box; }
  body { font-family: 'Malgun Gothic', sans-serif; background: #f4f5f7; color: #1f2328; margin: 0; padding: 24px; }
  .wrap { max-width: 960px; margin: 0 auto; }
  .panel { background: #fff; border: 1px solid #e2e4e8; border-radius: 10px; padding: 16px 20px; margin-bottom: 14px; }
  h1 { font-size: 18px; margin: 0 0 4px; }
  .sub { font-size: 12px; color: #6b7280; }
  .row { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; margin-top: 10px; }
  input, select { padding: 7px 10px; border: 1px solid #d1d5db; border-radius: 6px; font-family: inherit; font-size: 14px; }
  button { padding: 8px 18px; border: 1px solid #1d4ed8; border-radius: 6px; background: #1d4ed8; color: #fff; cursor: pointer; font-family: inherit; font-size: 14px; }
  button:hover { opacity: .9; }
  button.ghost { background: #fff; color: #1d4ed8; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; margin-top: 8px; }
  th, td { padding: 7px 10px; border-bottom: 1px solid #eef0f3; text-align: right; }
  th { background: #f8fafc; color: #6b7280; }
  td:first-child, th:first-child { text-align: left; }
  .on1 { color: #1d4ed8; font-weight: 700; }
  .good { color: #15803d; font-weight: 600; }
  .bad { color: #b45309; font-weight: 600; }
  .muted { color: #9ca3af; font-size: 12px; }
  .secTitle { font-size: 13px; color: #6b7280; font-weight: 600; margin: 0 0 4px; }
  .hidden { display: none; }
</style>
</head>
<body>
<div class="wrap">

  <div class="panel">
    <h1>지하철 승하차 수요예측 — 추론 대시보드</h1>
    <div class="sub">역 1104 · 모델 직접 추론 (models_v3.pkl) · 오늘부터 미래 아무 일시나 선택 가능</div>
    <div class="row">
      <label>날짜 <input type="date" id="d"></label>
      <label>시각 <select id="t"></select></label>
      <button class="ghost" onclick="calcFeatures()">1. 피처 계산</button>
      <button onclick="runInfer()">2. 추론 실행</button>
    </div>
    <div class="muted" style="margin-top:6px;">운행창 05:00 ~ 익일 00:10 (5분 단위) · 00:00~00:10은 전날 막차의 하차 꼬리</div>
  </div>

  <div class="panel hidden" id="featPanel">
    <p class="secTitle">파생피처 — 이 일시에서 모델에 들어가는 입력값 <span id="dowName" class="muted"></span></p>
    <table id="featTable"></table>
  </div>

  <div class="panel hidden" id="predPanel">
    <p class="secTitle">추론 결과 — 기준 시점 이후 각 윈도우의 승·하차 인원</p>
    <table id="predTable"></table>
  </div>

</div>

<script>
const $ = id => document.getElementById(id);

/* 시각 드롭다운: 05:00~23:55 + 00:00~00:10 (운행창) */
(function initTimes() {
  const sel = $('t'), opts = [];
  for (let h = 5; h < 24; h++)
    for (let m = 0; m < 60; m += 5)
      opts.push(String(h).padStart(2,'0') + ':' + String(m).padStart(2,'0'));
  opts.push('00:00', '00:05', '00:10');
  sel.innerHTML = opts.map(o => `<option ${o==='08:30'?'selected':''}>${o}</option>`).join('');
})();

fetch('/api/meta').then(r => r.json()).then(m => {
  const d = $('d');
  d.min = m.date_min; d.max = m.date_max; d.value = m.date_min;
});

function ts() { return $('d').value + ' ' + $('t').value + ':00'; }

async function calcFeatures() {
  const res = await fetch('/api/features?ts=' + encodeURIComponent(ts()));
  const r = await res.json();
  if (!res.ok) { alert(r.error); return; }
  $('dowName').textContent = '(' + r.dow_name + '요일)';
  $('featTable').innerHTML =
    '<tr><th>피처</th><th>설명</th><th>값</th></tr>' +
    r.features.map(f =>
      `<tr><td>${f.name}</td><td style="text-align:left;color:#6b7280;">${f.label}</td>
       <td class="${f.value>0 && f.name.startsWith('is')||f.name.startsWith('holiday') ? 'on1':''}">${f.value}</td></tr>`
    ).join('');
  $('featPanel').classList.remove('hidden');
  $('predPanel').classList.add('hidden');
}

async function runInfer() {
  const res = await fetch('/api/predict?ts=' + encodeURIComponent(ts()));
  const r = await res.json();
  if (!res.ok) { alert(r.error); return; }
  const rows = [['board','승차'], ['alight','하차']].map(([col, name]) => {
    const cells = [5, 15, 30, 60].map(w =>
      `<td>${r.pred[col + '_' + w].toLocaleString()}명</td>`).join('');
    return `<tr><td>${name}</td>${cells}</tr>`;
  }).join('');
  $('predTable').innerHTML =
    '<tr><th></th><th>+5분</th><th>+15분</th><th>+30분</th><th>+60분</th></tr>' + rows;
  $('predPanel').classList.remove('hidden');
}
</script>
</body>
</html>"""

if __name__ == '__main__':
    print(f'모델 {len(MODELS)}개 로드 (피처 {len(FEATURES)}개: {FEATURES})')
    print('http://localhost:5000 에서 확인하세요')
    app.run(host='127.0.0.1', port=5000, debug=False)
