# 🎯 문제 원인 최종 진단

## ✅ 데이터는 정상입니다!

### VitalDB 테스트 데이터 통계:
```
Propofol (mg/kg/h):
  Mean: 5.460 ± 3.046
  Median: 5.393
  Range: [0.000, 29.998]
  
Remifentanil (μg/kg/min):
  Mean: 0.1382 ± 0.1566
  Median: 0.1184
  Range: [0.0103, 7.4315]
```

✅ **정상 범위입니다!**
- Propofol 평균 5.46 mg/kg/h는 임상 정상 범위 (3-12)
- Remifentanil 평균 0.138 μg/kg/min은 임상 정상 범위 (0.05-0.4)

---

## 🚨 실제 문제: 모델 출력이 너무 작음

### Simulator 테스트에서 모델이 출력한 값:

**Quantum Agent:**
- Propofol: **0.500 mg/kg/h** (정상의 1/10)
- Remifentanil: **0.0099 μg/kg/min** (정상의 1/10)

**Classical Agent:**
- Propofol: **0.383 mg/kg/h** (정상의 1/13)
- Remifentanil: **0.0014 μg/kg/min** (정상의 1/100)

---

## 💡 MDAPE 90%의 의미가 이제 명확해짐

### VitalDB 테스트에서:
```python
# 실제 약물량 (VitalDB test)
True Propofol: 5.46 mg/kg/h
True Remifentanil: 0.138 μg/kg/min

# 모델 예측 (추정)
Predicted Propofol: ~0.5 mg/kg/h
Predicted Remifentanil: ~0.01 μg/kg/min

# MDAPE 계산
Propofol MDAPE = |5.46 - 0.5| / 5.46 * 100 = 91% ✓
Remifentanil MDAPE = |0.138 - 0.01| / 0.138 * 100 = 93% ✓
```

**결론:** MDAPE 91%는 정확한 측정값입니다. 모델이 **실제로 약 90% 틀린 약물량을 예측**하고 있습니다!

---

## 🔍 왜 모델이 10배 작은 값을 출력할까?

### 가설 1: Action Space Clipping
```python
# dual_drug_env.py Line 317
action = np.clip(action, self.action_space.low, self.action_space.high)
```
✅ Action space는 [15.0, 0.5]로 올바름 → 이건 문제 아님

### 가설 2: Actor Network Output Range

**일반적인 DDPG Actor:**
```python
def forward(self, state):
    x = self.net(state)
    return torch.tanh(x)  # 출력 범위: [-1, 1]
```

**Scaling:**
```python
action = (tanh_output + 1) / 2 * action_high
# tanh=0 → action = action_high / 2
# tanh=-1 → action = 0
# tanh=+1 → action = action_high
```

**문제:** tanh 출력이 -0.9 ~ -0.8 범위에 머물면:
- Propofol = (tanh + 1) / 2 * 15.0
- tanh = -0.93 → Propofol = 0.07 / 2 * 15 = 0.525 ✓ **이것입니다!**

---

## 📊 근본 원인 분석

### 1. **Offline Training 문제**

VitalDB 데이터에서 Behavioral Cloning:
- 모델이 의사의 행동 패턴을 학습
- 하지만 **offline loss가 제대로 수렴하지 않음**
- BC (Behavioral Cloning) 손실이 높으면 → Actor가 랜덤 초기화 상태 근처에 머묾

### 2. **Online Fine-tuning 실패**

Online fine-tuning (100 episodes):
- **충분하지 않음**
- Exploration noise가 너무 작음
- Actor의 tanh가 음수 영역(-1 근처)에서 벗어나지 못함

### 3. **Reward Signal 문제**

```python
# Time in Target: 0.0%
# Reward: -191
```

모델이 **어떤 episode에서도 성공 경험을 못함**
- → Positive reward를 한 번도 못 받음
- → Gradient가 나쁜 방향으로만 학습
- → Actor가 더 안전한 방향(약물 적게)으로 수렴

---

## ✅ 해결 방안 (우선순위 순)

### 🔥 1. 즉시 적용: Warmstart with Better Initialization

```python
# Actor 초기화 개선
class Actor(nn.Module):
    def __init__(self, ...):
        # ... 기존 코드 ...
        
        # 최종 레이어 bias를 양수로 초기화
        # tanh(0.5) ≈ 0.46 → action ≈ 0.73 * action_high
        nn.init.constant_(self.output_layer.bias, 0.5)
```

### 🔥 2. 즉시 적용: Action Scaling 확인

```python
# agents/quantum_agent.py, classical_agent.py
def select_action(self, state):
    action = self.actor(state)
    
    # tanh 출력 확인
    print(f"Raw actor output (before scaling): {action}")
    
    # Scaling 적용
    action = (action + 1.0) / 2.0 * self.action_high
    print(f"Scaled action: {action}")
    
    return action
```

### 🔥 3. Offline Training 개선

```python
# Behavioral Cloning weight 증가
bc_weight = 0.95  # 0.8 → 0.95

# 또는 supervised learning phase 추가
for epoch in range(10):
    # Pure BC (no RL)
    actor_loss = F.mse_loss(predicted_action, true_action)
```

### 🔥 4. Online Fine-tuning 강화

```python
# Episode 수 증가
online_episodes = 500  # 100 → 500

# Exploration 강화
exploration_noise_std = 2.0  # 0.1 → 2.0 (action scale에 맞춤)

# Warmup episodes 증가
warmup_episodes = 200  # 50 → 200
```

### 🔥 5. Curriculum Learning

```python
# Stage 2-1: High reward threshold
for episode in range(100):
    # BIS target: 40-60 (넓은 범위)
    # 약물 효율 페널티 제거
    
# Stage 2-2: Normal training
for episode in range(200):
    # BIS target: 45-55 (정상 범위)
    # 정상 reward 함수
```

---

## 🎯 수정 우선순위

### Phase 1 (즉시): Actor 초기화 수정
1. Actor 최종 레이어 bias를 양수로 초기화
2. Action scaling 로그 추가하여 확인

### Phase 2 (단기): Training 개선  
3. BC weight 증가 또는 supervised warmup 추가
4. Online episodes 증가 + exploration 강화

### Phase 3 (중기): Curriculum Learning
5. 2단계 학습: 넓은 target → 좁은 target

---

## 📈 예상 개선 효과

**현재:**
- MDAPE: 91% (10배 작은 약물량)
- Time in Target: 0%
- Reward: -191

**Phase 1 수정 후 예상:**
- MDAPE: 30-50% (3-5배 개선)
- Time in Target: 20-40%
- Reward: -50 ~ -100

**Phase 2 수정 후 예상:**
- MDAPE: 10-20%
- Time in Target: 50-70%
- Reward: -20 ~ -50

**Phase 3 수정 후 목표:**
- MDAPE: < 10%
- Time in Target: > 70%
- Reward: > -20
