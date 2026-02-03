# 단위 및 스케일링 분석 보고서

## 문제 요약

사용자가 propofol과 remifentanil의 최대값을 **30과 50에서 12와 2로 변경**했는데, 전체적인 스케일링과 단위가 맞는지 확인 필요.

## 현재 설정 값

### 환경 (dual_drug_env.py:218-219)
```python
self.action_space = spaces.Box(
    low=np.array([0.0, 0.0]),
    high=np.array([30.0, 1.0]),  # [mg/kg/h, μg/kg/min]
    dtype=np.float32
)
```
- **Propofol**: 0-30 mg/kg/h
- **Remifentanil**: 0-1.0 μg/kg/min

### 데이터 검증 (vitaldb_loader.py:360)
```python
if action < 0 or action > 30:  # Unrealistic propofol rates
    continue
```

## 실제 데이터 분석

### 데이터셋: `vitaldb_offline_data_small.pkl`
- **Total transitions**: 18,420

### Propofol 분석
```
[4] Previous propofol action (mg/kg/h):
    Range: [0, 1487.77]  ⚠️ 문제!
    Mean: 131.90
    90th percentile: 205.22
    
[6] Propofol accumulation (sum of 7 timesteps):
    Range: [-1.95, 1.88]
    Mean: 0.01
```

### Remifentanil 분석
```
[5] Previous remifentanil action (μg/kg/min):
    Range: [0, 0]  ⚠️ 데이터 없음!
    
[7] Remifentanil accumulation:
    Range: [0, 0]
```

## ⚠️ 발견된 문제들

### 1. **Propofol 단위 불일치** (CRITICAL)

**라인 233**: 단위 변환 코드
```python
# PPF20_RATE: mL/hr (20 mg/mL) → mg/kg/h
df['PPF20_RATE'] = df['PPF20_RATE'] * 20.0 / patient_weight
```

**문제**: `[4] Previous propofol action`의 최대값이 **1487.77 mg/kg/h**로 비정상적
- 임상적으로 불가능한 값 (일반적 최대: 12 mg/kg/h)
- 라인 360의 검증(`> 30`)을 통과하지만 여전히 비현실적

**원인 추정**: 
- 단위 변환 오류
- 또는 누적(accumulation) vs 순간 rate의 혼동

### 2. **Remifentanil 데이터 부재**

모든 remifentanil 값이 0으로, 실제로는 single-drug (propofol only) 데이터셋

### 3. **Accumulation 값의 단위**

**라인 965-967**: Accumulation 계산
```python
start_idx = max(0, idx - 6)
ppf_acc = df.iloc[start_idx:idx+1]['PPF20_RATE'].sum()  # 7개 합
rftn_acc = df.iloc[start_idx:idx+1]['RFTN_RATE'].fillna(0).sum()
```

**문제**: 
- `PPF20_RATE`는 이미 `mg/kg/h` 단위
- 7개 timesteps (70초) 합산 → 단위가 `(mg/kg/h) × 7` = 무의미한 단위
- **올바른 방법**: rate를 시간으로 곱한 후 합산해야 함

**수정 방안**:
```python
# 각 timestep은 10초 = 10/3600 시간
timestep_hours = 10.0 / 3600.0
ppf_acc = df.iloc[start_idx:idx+1]['PPF20_RATE'].sum() * timestep_hours
# → 단위: mg/kg
```

## 이론적 최대값 계산

### 변경 제안: 12 mg/kg/h (propofol), 2 μg/kg/min (remifentanil)

#### Propofol Accumulation
- **최대 rate**: 12 mg/kg/h
- **시간**: 7 timesteps × 10초 = 70초 = 70/3600 시간 ≈ 0.0194 시간
- **이론적 최대**: `12 mg/kg/h × 0.0194 h = 0.233 mg/kg`

**현재 데이터**: 
- Max accumulation = 1.88
- **문제**: 1.88은 무단위 (잘못된 계산)

#### Remifentanil Accumulation
- **최대 rate**: 2 μg/kg/min  
- **시간**: 70초 = 70/60 분 ≈ 1.167 분
- **이론적 최대**: `2 μg/kg/min × 1.167 min = 2.33 μg/kg`

## 📋 권장 수정사항

### 우선순위 1: Accumulation 계산 수정

**파일**: `src/data/vitaldb_loader.py:965-967`

```python
# 현재 (잘못됨)
ppf_acc = df.iloc[start_idx:idx+1]['PPF20_RATE'].sum()

# 수정안
timestep_seconds = 10.0
ppf_acc = df.iloc[start_idx:idx+1]['PPF20_RATE'].sum() * (timestep_seconds / 3600.0)  # mg/kg
rftn_acc = df.iloc[start_idx:idx+1]['RFTN_RATE'].fillna(0).sum() * (timestep_seconds / 60.0)  # μg/kg
```

### 우선순위 2: Action 범위 검증 강화

**파일**: `src/data/vitaldb_loader.py:360`

```python
# 현재
if action < 0 or action > 30:  # Unrealistic propofol rates

# 수정안 (12 mg/kg/h로 제한)
if action < 0 or action > 12:  # Clinical maximum for propofol
    continue
```

### 우선순위 3: 환경 action space 업데이트

**파일**: `src/environment/dual_drug_env.py:218-219`

```python
# 현재
high=np.array([30.0, 1.0]),  # [mg/kg/h, μg/kg/min]

# 수정안 (임상 기준에 맞춤)
high=np.array([12.0, 2.0]),  # [mg/kg/h, μg/kg/min]
```

### 우선순위 4: Normalization 일관성

**파일**: `src/data/vitaldb_loader.py:377`

```python
# 현재
'actions': np.array(actions_list, dtype=np.float32).reshape(-1, 1) / action_max,

# action_max도 12.0으로 변경 필요
```

## 임상 참고값

### Propofol
- **유도(Induction)**: 1.5-2.5 mg/kg bolus
- **유지(Maintenance)**: 4-12 mg/kg/h
- **최대 권장**: 12 mg/kg/h (장기간 사용시 더 낮음)

### Remifentanil  
- **유도**: 0.5-1 μg/kg bolus
- **유지**: 0.05-0.3 μg/kg/min (일반)
- **최대**: 0.5-2 μg/kg/min (수술 상황에 따라)

## 결론

### 30 → 12, 50 → 2 변경의 의미

**질문의 "50"은 오해로 추정됩니다**:
- 환경 설정에는 remifentanil 최대값이 **1.0** μg/kg/min
- "50"은 아마도 **50 μg/mL 농도**를 말하는 것으로 보임

**올바른 최대값**:
- ✅ Propofol: **12 mg/kg/h** (임상적으로 적절)
- ✅ Remifentanil: **2 μg/kg/min** (임상 최대 범위)

### 단위 일관성 체크

| 항목 | 현재 단위 | 올바른 단위 | 상태 |
|------|----------|------------|------|
| PPF20_RATE (변환 후) | mg/kg/h | mg/kg/h | ✅ |
| RFTN_RATE (변환 후) | μg/kg/min | μg/kg/min | ✅ |
| ppf_acc (계산) | 무단위 (잘못됨) | mg/kg | ❌ |
| rftn_acc (계산) | 무단위 (잘못됨) | μg/kg | ❌ |
| action space | mg/kg/h, μg/kg/min | mg/kg/h, μg/kg/min | ✅ (범위만 수정 필요) |

## 다음 단계

1. ✅ **Accumulation 계산 수정** (timestep duration 고려)
2. ✅ **Action space 범위 업데이트** (12, 2로)
3. ✅ **데이터 검증 임계값 조정** (30 → 12)
4. ✅ **action_max 매개변수 업데이트**
5. 🔄 **데이터 재생성** (수정된 로더로)
6. 🔄 **모델 재학습** (새 데이터로)
