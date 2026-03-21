# TD Estimator Selection Plan for Diffusion Planner Guidance

**Context**: Goal-conditioned value function V(s,g) = ψ(s)ᵀφ(g) (dual representation, arXiv:2510.06714)
used as TD estimator to provide gradient guidance (∇_s V) for a diffusion planner.

---

## 1. 문제 정의

### 1.1 Gradient Saturation 문제

standard discounted GCVF의 이론적 형태:

```
V*(s,g) = -(1 - γ^{d*(s,g)}) / (1 - γ)
```

γ=0.995, d*=100 → V* ≈ -39.4
γ=0.995, d*=200 → V* ≈ -63.2
γ=0.995, d*=100 vs d*=101 → ΔV ≈ **0.04** (거의 동일)

인접 state 간의 V 차이가 지수적으로 작아지기 때문에 goal로부터 먼 state에서
∇_s V(s,g) 는 신호 대 잡음비가 낮아져 diffusion planner의 guidance로 사용할 수 없다.

### 1.2 현재 코드 구조

- **Phase 1** (`train_dual_ogbench.py`): V = ψ(s)ᵀφ(g), Q = MLP(s,a,g) 별도 학습
- **Phase 2** (`train_gcvf_dual_ogbench.py`): V_down = GoalConditionedValue([s, φ(g)]), IQL loss
- **Guidance 사용 예정**: ∇_s V_down(s, φ(g)) 또는 ∇_s (ψ(s)ᵀφ(g))

---

## 2. 사용자 제안 전략 평가

### 전략 1: γ = 1 (선형 감소)

**아이디어**: V*(s,g) = -d*(s,g) (temporal distance). 지수 감쇠 없이 선형이므로 gradient scale 거리 무관.

**이론 분석**:
- 장점: ΔV = 1 (거리와 무관), gradient scale 균등
- 문제: Bellman operator T^π는 γ-contraction. **γ=1에서 contraction 소실 → TD 발산**
  - Tsitsiklis & Van Roy (1996): function approximation + off-policy + γ=1 = deadly triad 모두 충족 → 발산 보장
  - OGBench 환경은 offline 데이터 = off-policy, function approximation 사용 → 발산 조건 해당

**평가**: 이론적으로 매력적이나 off-policy offline 세팅에서 TD 학습으로는 **구현 불가**
(QRL, CRL 등 γ=1에 해당하는 방법들은 모두 **TD bootstrapping을 사용하지 않음**)

---

### 전략 2: γ≈1 유지 + 추론 시 low-pass filter (mean pooling)

**아이디어**: V(s,g)에 공간적 평균을 씌워 gradient 방향 안정화.

```
V_σ(s) = E_{ε~N(0,σ²I)}[V(s+ε)]
∇_s V_σ = (1/σ²) E_ε[ε · V(s+ε)]   (Stein's identity)
```

**이론 분석**:
- σ 작을 때: 원래 gradient와 동일 → saturation 해결 안 됨
- σ 클 때: 공간적으로 블러된 V의 gradient → **편향(bias) 발생**
  완전히 평탄한 V 구간에서 σ를 키우면 gradient가 가장 가까운 "높은 V" 지역을 가리킬 수 있으나, 이는 지정 goal g로의 방향이 아닐 수 있음
- 최적 σ를 결정하려면 d*(s,g)를 알아야 하는 순환 문제

**관련 연구**: Guo et al. (NeurIPS 2024, arXiv:2404.14743) "Gradient Guidance for Diffusion Models"에서
Tweedie 기반 look-ahead 제안: V(x̂_0|x_t) 사용 (단순 mean pooling보다 원칙적인 대안)

**평가**: 원리는 타당하나 σ 튜닝이 까다롭고 편향 제어가 어려움.
더 원칙적인 변형(Tweedie look-ahead)이 존재하므로 직접 구현보다 해당 방법 채택 권장.

---

### 전략 3: 학습 과정에서 인접 state 간 상대적 value 차이 보존 trick

**아이디어**: 학습 시 V(s_t, g)와 V(s_{t+1}, g)의 차이가 잘 보존되도록 유도.

**관련 연구**:
1. **TRL** (Park et al., ICLR 2026, arXiv:2510.22512): 삼각부등식 V(s,g) ≥ V(s,w)+V(w,g)를 활용한 divide-and-conquer
2. **n-step returns** (Park et al., 2025, arXiv:2506.04168): "Q errors become larger as distance from end increases. 64-step returns are nearly flat vs 1-step."
3. **TMD** (Myers et al., 2025): Lipschitz quasimetric 제약 → |V(s,g) - V(s',g)| ≤ d*(s,s') 보장

**평가**: **현재 코드에 가장 쉽게 통합 가능한 전략**. n-step returns은 drop-in 개선이며
TRL-style transitive loss는 현재 IQL 목적함수에 추가 항으로 병행 가능.
단, 근본적인 지수 포화를 완전히 해소하지는 못하고 bias를 줄이는 수준.

---

## 3. 대안 전략 (선행 연구 기반)

### 전략 4: Log-Space Value (CRL / CMD) — 🌟 가장 근본적 해결

**핵심 아이디어**: V 대신 **log p(reach g | from s)**를 학습. Log-space에서 gradient scale이 거리 무관.

**수학적 증명**:
```
V(s,g) = γ^{d*} / (1-γ)    →  ∇_s V = log(γ) · γ^{d*}·∇_s d*   (→ 0 as d*→∞)
f(s,g) = log(γ^{d*}) = d*·log(γ)  →  ∇_s f = log(γ)·∇_s d*       (거리 무관!)
```

**주요 논문**:
- **CRL**: Eysenbach et al., "Contrastive Learning as Goal-Conditioned RL," NeurIPS 2022 (arXiv:2206.07568)
  학습 방법: InfoNCE, critic f(s,a,g) = φ(s,a)ᵀψ(g), 수렴 시 = log p(reach g | s,a) + c
- **CMD**: Myers et al., "Learning Temporal Distances: Contrastive Successor Features," ICML 2024 (arXiv:2406.17098)
  추가로 quasimetric 제약 적용 → stochastic MDP에서도 삼각부등식 만족
- **Stabilizing CRL**: Eysenbach et al., ICLR 2024 Spotlight (arXiv:2306.03346), 2× performance

**현재 구조와의 호환성**:
- Inner product 표현 (ψ(s)ᵀφ(g))은 유지 가능 → CRL objective로 교체
- OGBench에 contrastive 방식 적용 가능 (offline trajectory 데이터 활용)
- **주의**: InfoNCE는 large batch 요구 (B ≥ 1024). 현재 B=2048 ✓

**장점**: gradient saturation 근본 해결, TD bootstrapping 불필요
**단점**: Phase 1 학습 objective 전면 교체 필요, Phase 2도 영향받음

---

### 전략 5: HILP L2 Distance 표현으로 전환 — 🌟 현재 구조 최소 변경

**핵심 아이디어**: 현재 V = ψ(s)ᵀφ(g) (inner product) →  V = -||ψ(s) - φ(g)|| (L2 distance)로 변경

**Gradient 분석**:
```
V(s,g) = -||ψ(s) - φ(g)||
∇_s V = -(ψ(s) - φ(g)) / ||ψ(s) - φ(g)|| · Jψ(s)
```

방향: **(ψ(s) - φ(g)) / ||...||** → latent space에서 goal을 향하는 **단위 벡터**
크기: Jψ(s) (embedding Jacobian)에만 의존, **d*(s,g)와 무관**

즉, distance normalization이 saturation 문제를 구조적으로 해결.
HILP 원논문 (arXiv:2402.15567)이 이 방식 사용.

**현재 코드 변경 범위**:
- `DualGoalPhiValue.__call__`: inner product → L2 distance
- IQL loss: V 범위가 (-∞, 0)으로 변경되므로 reward scale 확인 필요
- Phase 2 GCVF는 그대로 유지 가능 (φ(g) 고정 후 MLP 학습)

**장점**: 코드 최소 변경, gradient 방향이 구조적으로 항상 goal-directed
**단점**: Inner product의 이론적 속성(dual representation) 일부 손실

**관련 논문**: Park et al., ICML 2024 "Foundation Policies with Hilbert Representations" (arXiv:2402.15567)

---

### 전략 6: Quasimetric RL (QRL) — TD 없이 d*(s,g) 직접 학습

**논문**: Wang et al., "Optimal Goal-Reaching RL via Quasimetric Learning," ICML 2023 (arXiv:2304.01203)

**방법**: Dual optimization으로 temporal distance 직접 학습 (bootstrapping 없음)
Töpperwien et al. (ICML 2026, arXiv:2602.05459): QRL vs HIQL 비교에서 QRL이 gradient interference 현저히 낮음

**장단점**: 가장 원칙적인 해결이나 **아키텍처 전면 교체** 필요.

---

### 전략 7: Hierarchical Guidance (HIQL-style) — Goal에서 가까운 subgoal 사용

**논문**: Park et al., "HIQL: Offline Goal-Conditioned RL with Latent States as Actions," NeurIPS 2023 (arXiv:2307.11949)
Chen et al., "Simple Hierarchical Planning with Diffusion," ICLR 2024 (arXiv:2401.02644)

**방법**: 멀리 있는 g 대신 H-step 앞의 subgoal w를 예측하여 V(s,w)의 gradient 사용
W는 s에서 가깝기 때문에 V(s,w)가 비교적 정확하고 gradient가 정보를 가짐

**현실적 평가**: Diffusion planner에서 subgoal hierarchical guidance는
Simple Hierarchical Planning with Diffusion 논문이 정확히 이 아이디어를 구현하여
Maze2D +12%, MuJoCo +9.2% 성능 향상 달성.

---

### 전략 8: Derivative-Free Guidance (SVDD) — gradient 계산 회피

**논문**: Li et al., "Derivative-Free Guidance via Soft Value-Based Decoding," 2024 (arXiv:2408.08252)

**방법**: 각 denoising step에서 K개 후보를 sampling, soft value function으로 reweighting:
```
p*(x_{t-1}|x_t) ∝ p^pre(x_{t-1}|x_t) · exp(V(x̂_0(x_{t-1})) / α)
```

**평가**: Gradient saturation 문제 원천 회피. 단, K배 계산 비용.

---

### 전략 9: n-Step Returns (가장 쉬운 drop-in 개선)

**논문**: Park et al., "Horizon Reduction Makes RL Scalable," 2025 (arXiv:2506.04168)

```
R_t^n = Σ_{i=0}^{n-1} γ^i r_{t+i} + γ^n V(s_{t+n})
```

n=64 step return은 1-step TD 대비 Q-error를 거리 전반에 걸쳐 **균등하게 유지**
(논문에서 직접 측정하여 보고). 현재 IQL 코드에 최소 수정으로 적용 가능.

---

## 4. 종합 비교표

| 전략 | Saturation 해결 | TD 안정성 | 코드 변경 범위 | 계산 비용 | 추천도 |
|------|----------------|-----------|----------------|-----------|--------|
| 1. γ=1 (TD) | ✓ (이론) | ✗ 발산 | 최소 | 동일 | ✗ |
| 2. Low-pass filter | △ 부분적 | N/A | 추론부만 | 저 | △ |
| 3. n-step returns | △ 편향 감소 | ✓ | 최소 | 동일 | ○ |
| **4. Log-space (CRL/CMD)** | **✓ 근본 해결** | N/A (no TD) | **Phase 1 전면** | 중간 | **◎** |
| **5. L2 distance (HILP)** | **✓ 구조적 해결** | ✓ | **최소** | 동일 | **◎** |
| 6. QRL (quasimetric) | ✓ | N/A | 전면 교체 | 중간 | △ |
| 7. Hierarchical subgoal | ✓ (간접) | ✓ | 중간 | 중간 | ○ |
| 8. Derivative-free (SVDD) | ✓ (회피) | N/A | 추론부만 | K배↑ | ○ |
| **3+5 조합** | **✓** | **✓** | **최소** | **동일** | **최고** |

---

## 5. 최종 권장안: 전략 5 (L2 Distance) + 전략 3 (n-step Returns)

### 이유

현재 코드 맥락에서:
1. **Phase 1은 ψ(s)ᵀφ(g) inner product** → V = -||ψ(s) - φ(g)||로 바꾸면
   **gradient 방향이 구조적으로 goal-directed 단위벡터**가 되어 saturation 문제 해결
2. **Phase 2 GCVF는 영향 없음** (φ(g) 고정 후 MLP 학습, gradient는 Phase 1에서만 사용)
3. **n-step returns은 코드 최소 수정**으로 먼 goal에서의 V 추정 정확도 향상

만약 이 조합이 불충분하다면, **CRL log-space objective로 Phase 1 재학습**이 가장 원칙적인 해결.

### 구현 시 주의사항

**L2 distance 전환 시**:
```python
# DualGoalPhiValue.__call__ 변경
def __call__(self, observations, goals=None):
    psi_s = self.psi(observations)    # (B, D)
    phi_g = self.phi(goals)           # (B, D)
    # 기존: v = (psi_s * phi_g).sum(axis=-1)
    squared_dist = ((psi_s - phi_g) ** 2).sum(axis=-1)
    v = -jnp.sqrt(jnp.maximum(squared_dist, 1e-6))  # (B,)
    return v
```

- V 범위: (-∞, 0), reward normalization 유지
- Temperature-scaled sigmoid 불필요 (gradient가 이미 unit vector direction으로 제한됨)
- Q network는 현행 유지

**n-step return 전환 시**:
`GCDataset`의 return 계산 부분을 multi-step으로 교체 (n=8~64).

---

## 6. Temperature-Scaled Sigmoid 관련 정정

사용자 질문에서 Q와 V 모두에 -sigmoid(f/T)를 적용하는 방안을 검토했으나:

**L2 distance 방식 채택 시**: V = -||ψ-φ|| 자체가 (-∞, 0)으로 이미 bounded below.
gradient 방향이 unit vector이므로 bounded output을 위한 activation 불필요.
(Inference 시 gradient normalization + guidance scale α 조합으로 충분)

**Inner product 방식 유지 시**: Phase 2 GCVF에만 -sigmoid(f/T) 적용 권장.
Phase 1 Q network에는 적용하지 않음 (Q가 reward의 bellman equation target이므로 자유로운 범위 필요).

---

## 참고 문헌

1. Wang et al., "Optimal Goal-Reaching RL via Quasimetric Learning," ICML 2023. arXiv:2304.01203
2. Eysenbach et al., "Contrastive Learning as Goal-Conditioned RL," NeurIPS 2022. arXiv:2206.07568
3. Myers et al., "Learning Temporal Distances: Contrastive Successor Features," ICML 2024. arXiv:2406.17098
4. Park et al., "Foundation Policies with Hilbert Representations (HILP)," ICML 2024. arXiv:2402.15567
5. Park et al., "HIQL: Offline Goal-Conditioned RL with Latent States as Actions," NeurIPS 2023. arXiv:2307.11949
6. Park et al., "Transitive RL: Value Learning via Divide and Conquer," ICLR 2026. arXiv:2510.22512
7. Park et al., "Horizon Reduction Makes RL Scalable," 2025. arXiv:2506.04168
8. Li et al., "Derivative-Free Guidance via Soft Value-Based Decoding (SVDD)," 2024. arXiv:2408.08252
9. Guo et al., "Gradient Guidance for Diffusion Models: Optimization Perspective," NeurIPS 2024. arXiv:2404.14743
10. Myers et al., "Offline Goal-Conditioned RL with Temporal Distance Representations (TMD)," 2025.
11. Chen et al., "Simple Hierarchical Planning with Diffusion," ICLR 2024. arXiv:2401.02644
12. Jackson et al., "Policy-Guided Diffusion," RLC 2024. arXiv:2404.06356
13. Psenka et al., "Q-Score Matching," ICML 2024. arXiv:2312.11752
14. Farebrother et al., "Stop Regressing: Value Functions via Classification (HL-Gauss)," ICML 2024. arXiv:2403.03950
15. Pong et al., "Temporal Difference Models (TDM)," ICLR 2018. arXiv:1802.09081
16. Töpperwien et al., "When Are RL Hyperparameters Benign?," ICML 2026. arXiv:2602.05459
17. Tsitsiklis & Van Roy, "An Analysis of Temporal-Difference Learning with Function Approximation," IEEE TAC, 1997.
18. Eysenbach et al., "Stabilizing Contrastive RL," ICLR 2024 Spotlight. arXiv:2306.03346
