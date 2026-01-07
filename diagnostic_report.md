# 🔍 문제 진단 보고서

## 실험 결과 요약

- **VitalDB 테스트**: Propofol MDAPE 91.55%, Remifentanil MDAPE 91.72%
- **Simulator 테스트**: MDAPE 95.66%
- **Time in Target**: 0.0%
- **Reward**: -191 (매우 낮음)
- **Propofol 사용량**: 0.38-0.50 mg/kg/h (정상: 3-12 mg/kg/h)
- **Remifentanil 사용량**: 0.001-0.01 μg/kg/min (정상: 0.05-0.4 μg/kg/min)

---

## 🚨 핵심 문제점

### 1. **MDAPE 측정 대상 오해**

**현재 코드:**
```python
# Line 503-506 in compare_quantum_vs_classical_dualdrug.py
ppf_error = np.abs(predicted_actions[:, 0] - actions[:, 0]) / (np.abs(actions[:, 0]) + 1e-6)
rftn_error = np.abs(predicted_actions[:, 1] - actions[:, 1]) / (np.abs(actions[:, 1]) + 1e-6)
```

- MDAPE는 **약물 투여량(action) 예측 오차**를 측정
- BIS 값 예측 오차가 아님!
- 따라서 MDAPE 90%는 "약물 투여량을 90% 틀렸다"는 의미

**문제:** Remifentanil 투여량이 매우 작을 때 (0.01 μg/kg/min 등) 작은 절대 오차도 큰 퍼센트 오차로 나타남

예시:
- 실제값: 0.01 μg/kg/min
- 예측값: 0.001 μg/kg/min  
- 오차: |0.01 - 0.001| / 0.01 = 90%

---

### 2. **약물 투여량이 비정상적으로 낮음**

**관찰된 값:**
- Propofol: 0.38-0.50 mg/kg/h
- Remifentanil: 0.001-0.01 μg/kg/min

**정상 범위:**
- Propofol: 3-12 mg/kg/h (약 **10배 부족**)
- Remifentanil: 0.05-0.4 μg/kg/min (약 **5-50배 부족**)

**원인:**
1. Action space는 올바르게 설정됨:
   ```python
   # Line 213-216 in dual_drug_env.py
   self.action_space = spaces.Box(
       low=np.array([0.0, 0.0]),
       high=np.array([15.0, 0.5]),  # 올바른 범위
       dtype=np.float32
   )
   ```

2. 하지만 **모델이 학습 중 낮은 값을 출력하도록 수렴**
   - Reward 신호가 약물 사용을 과도하게 페널티
   - 또는 VitalDB 학습 데이터의 스케일 문제

---

### 3. **Reward 함수 분석**

**Potential-based reward (Line 429-481):**
```python
# R_eff: Drug efficiency penalty
normalized_ppf = ppf_rate / 12.0
normalized_remi = rftn_rate / 0.3
r_eff = -(normalized_ppf + normalized_remi)

# Base reward
r_base = w1 * r_track + w2 * r_safe + w3 * r_eff  # w3=0.1
```

**문제:**
- `r_eff`는 항상 음수 (약물 사용량을 페널티)
- 약물을 적게 쓸수록 페널티가 작아짐
- 하지만 w3=0.1로 작은 가중치이므로 큰 문제는 아님

**관찰된 Reward: -191**
- 매우 낮은 값
- BIS tracking reward (r_track)가 매우 나쁨을 의미
- 모델이 목표 BIS에 도달하지 못함

---

### 4. **Time in Target 0%**

**의미:**
- BIS가 45-55 범위에 **단 한 번도** 들어가지 않음
- 모델이 완전히 실패

**예상 원인:**
- 약물 투여량이 너무 적어서 BIS를 낮추지 못함
- 또는 반대로 BIS가 너무 낮아서 범위를 벗어남

---

## 🔬 데이터 흐름 분석

### VitalDB 데이터 전처리

**Unit conversion (Line 238-245 in vitaldb_loader.py):**
```python
# PPF20_RATE: mL/hr (20 mg/mL) → mg/kg/h
df['PPF20_RATE'] = df['PPF20_RATE'] * 20.0 / patient_weight

# RFTN_RATE: mL/hr (20 mcg/mL) → μg/kg/min
df['RFTN_RATE'] = df['RFTN_RATE'] * 20.0 / patient_weight / 60.0
```

✅ 단위 변환은 올바름

### State representation (13D)

**Extended state (Line 638-673 in vitaldb_loader.py):**
```python
state = [
    bis_error,       # BIS - 50
    ce_ppf,          # Propofol Ce
    ce_rftn,         # Remifentanil Ce
    dbis_dt,         # BIS 변화율
    u_ppf_prev,      # 이전 propofol 투여량
    u_rftn_prev,     # 이전 remifentanil 투여량
    ppf_acc,         # 누적 propofol (1분)
    rftn_acc,        # 누적 remifentanil (1분)
    bis_slope,       # BIS 경향 (3분)
    age,             # 환자 나이
    sex,             # 환자 성별 (0/1)
    bmi,             # 환자 BMI
    bis_error ** 2   # BIS 오차 제곱
]
```

⚠️ **스케일 불일치 가능성:**
- `bis_error`: -20 ~ 20
- `ce_ppf`: 0 ~ 10 mcg/mL
- `age`: 20 ~ 80
- `bmi`: 15 ~ 40
- `ppf_acc`: 0 ~ 100+

**문제:** 서로 다른 스케일의 features가 정규화 없이 사용됨

---

## 🎯 근본 원인

### **핵심 문제: Offline 학습 데이터와 Online 환경의 불일치**

1. **VitalDB 데이터에서 학습:**
   - 실제 마취과 의사의 약물 투여 패턴 학습
   - Behavioral cloning으로 의사의 행동을 모방
   
2. **문제:**
   - VitalDB에서 remifentanil 사용량이 매우 적은 케이스가 많음
   - 또는 데이터 필터링 과정에서 remifentanil > 0.01만 선택 (Line 592)
   - 하지만 일부 케이스는 여전히 매우 낮은 값

3. **결과:**
   - 모델이 "적은 약물 = 안전"으로 학습
   - Online fine-tuning이 이를 극복하지 못함

---

## 📊 검증 필요사항

### 1. VitalDB 학습 데이터 분포 확인
```python
# 데이터 통계
print(f"Propofol rate: {actions[:, 0].mean():.3f} ± {actions[:, 0].std():.3f}")
print(f"Remifentanil rate: {actions[:, 1].mean():.3f} ± {actions[:, 1].std():.3f}")
print(f"Propofol range: [{actions[:, 0].min():.3f}, {actions[:, 0].max():.3f}]")
print(f"Remifentanil range: [{actions[:, 1].min():.3f}, {actions[:, 1].max():.3f}]")
```

### 2. 모델 출력값 확인
```python
# 실제로 모델이 출력하는 action 값
print(f"Model output range: [{predicted_actions.min():.3f}, {predicted_actions.max():.3f}]")
print(f"Model output mean: {predicted_actions.mean():.3f}")
```

### 3. BIS 값 추적
```python
# Simulator에서 BIS가 어떻게 변하는지
print(f"BIS trajectory: {bis_history}")
print(f"BIS mean: {np.mean(bis_history):.1f}")
print(f"BIS range: [{np.min(bis_history):.1f}, {np.max(bis_history):.1f}]")
```

---

## ✅ 해결 방안

### 1. **MDAPE 계산 수정 (단기)**

Action MDAPE 대신 **BIS tracking error**를 주 평가 지표로 사용:

```python
# 현재 (Action MDAPE)
mdape = np.median(np.abs(pred_action - true_action) / true_action) * 100

# 개선 (BIS tracking error)
bis_mae = np.mean(np.abs(bis_history - target_bis))
bis_in_target = np.mean((bis_history >= 45) & (bis_history <= 55)) * 100
```

### 2. **데이터 전처리 개선 (중기)**

**State normalization 추가:**
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
states_normalized = scaler.fit_transform(states)
```

**Action clipping 강화:**
```python
# 최소 약물량 제한
min_ppf = 2.0  # mg/kg/h
min_rftn = 0.03  # μg/kg/min

action[0] = np.clip(action[0], min_ppf, 15.0)
action[1] = np.clip(action[1], min_rftn, 0.5)
```

### 3. **Reward 함수 조정 (중기)**

```python
# 약물 효율 페널티 감소 또는 제거
w3 = 0.01  # 0.1 → 0.01 (약물 사용 페널티 감소)

# 또는 최소 약물량 이하일 때 추가 페널티
if ppf_rate < 2.0 or rftn_rate < 0.03:
    r_eff -= 1.0  # 너무 적은 약물 사용 페널티
```

### 4. **학습 데이터 필터링 (장기)**

```python
# VitalDB 데이터 필터링 강화
df_filtered = df[
    (df['BIS'] >= 40) & (df['BIS'] <= 60) &  # 좁은 BIS 범위
    (df['PPF20_RATE'] >= 3.0) &  # 최소 propofol
    (df['RFTN_RATE'] >= 0.05)  # 최소 remifentanil
]
```

### 5. **Offline → Online transition 개선 (장기)**

```python
# Online fine-tuning 시 exploration 강화
exploration_noise_std = 2.0  # 증가
warmup_episodes = 100  # 증가

# Curriculum learning
# 1단계: 높은 약물량에서 시작
# 2단계: 점진적으로 효율성 개선
```

---

## 🏁 결론

### 현재 상태
- ❌ 모델이 약물을 너무 적게 투여
- ❌ BIS 목표에 도달하지 못함
- ⚠️ MDAPE 90%는 약물량 예측 오차이므로 BIS 제어 성능과 직접 관련 없음

### 우선순위
1. **즉시**: 평가 지표를 BIS tracking error로 변경
2. **단기**: State normalization 추가
3. **중기**: Reward 함수 조정 + Action clipping
4. **장기**: VitalDB 데이터 필터링 개선

### 성공 기준
- Time in Target > 70%
- BIS MAE < 5
- Propofol 사용량: 4-10 mg/kg/h
- Remifentanil 사용량: 0.05-0.3 μg/kg/min
