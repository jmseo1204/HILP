# Dual Goal Representations 구현 계획

> 논문: [Dual Goal Representations (arXiv:2510.06714)](https://arxiv.org/abs/2510.06714)
> 저자: Seohong Park et al.
> 대상 코드베이스: `hilp_gcrl/`

---

## 1. hilp_gcrl vs hilp_zsrl 역할 차이

| 항목 | hilp_gcrl | hilp_zsrl |
|------|-----------|-----------|
| **문제 설정** | Goal-Conditioned RL (GCRL) | Zero-Shot RL (ZSRL) |
| **목표 정의** | 데이터셋 관측값으로 명시적 목표 지정 | 환경 보상함수로 암묵적 목표 정의 |
| **프레임워크** | JAX + Flax | PyTorch |
| **환경** | D4RL (AntMaze, Kitchen) | DeepMind Control Suite (DMC) |
| **φ의 역할** | 목표 도달 거리 계산용 Hilbert 표현 | 후계 특징 학습 (Successor Features) |
| **가치함수** | V(s,g) = −\|\|φ(s) − φ(g)\|\| | Q = z^T φ(s) (선형 보상 분해) |
| **스킬 학습** | φ-공간에서 방향 벡터로 스킬 정의 | z (goal feature)로 정책 조건화 |
| **평가** | 목표 관측값 도달 성공률 | 태스크 점수 (e.g., Walker 달리기) |

**핵심 차이**: `hilp_gcrl`은 φ를 공유 인코더(상태와 목표 동일 인코더)로 사용하여 V(s,g) = −‖φ(s) − φ(g)‖를 계산. `hilp_zsrl`은 φ를 feature learner로 두고 후계 특징 Q = z^T φ(s) 형태로 zero-shot task transfer를 지원.

---

## 2. Dual Goal Representations 논문 핵심 아이디어

### 2.1 기존 GCRL의 한계
- 기존 GCRL은 목표를 **원시 관측값(raw observation)** 그대로 사용
- 관측값에 외생적 노이즈(exogenous noise)가 있으면 목표 표현이 오염됨
- 태스크와 무관한 정보가 섞여 일반화 성능 저하

### 2.2 Dual Goal Representation의 정의
목표 g의 **Dual 표현**은 "모든 상태 s에서 g까지의 temporal distance 집합"으로 정의:

```
∨(g) : s ↦ d*(s, g)
```

즉, 목표를 **그 목표까지의 도달 난이도 프로파일**로 인코딩.

**핵심 성질**:
1. **충분성(Sufficiency)**: dual 표현만으로 최적 목표 도달 정책 복원 가능
2. **노이즈 불변성(Noise Invariance)**: 외생적 노이즈에 invariant
3. **다이나믹스 기반**: 원시 관측값이 아닌 환경 다이나믹스에만 의존

### 2.3 실용적 구현 (두 단계)

**Phase 1 — Temporal Distance 학습**
```
V(s, g) ≈ d*(s, g) = log_γ V*(s, g)

구현: f(ψ(s), φ(g)) = ψ(s)^T φ(g)
  - ψ: 상태 인코더 (state encoder)
  - φ: 목표 인코더 (goal encoder)
  - 내적(inner product)으로 값 계산
```

손실함수: 목표 조건 IQL expectile loss (현재 hilp_gcrl과 유사)

**Phase 2 — 추출된 φ(g)를 Goal Representation으로 사용**
```
학습된 φ(g)를 downstream GCRL 알고리즘의 goal input으로 대체
- 원시 관측값 g 대신 φ(g) ∈ R^d 사용
- GCIVL, CRL, GCFBC 등과 결합 가능
```

### 2.4 HILP과의 차이점

| 항목 | 기존 HILP (hilp_gcrl) | Dual Goal Repr. |
|------|-----------------------|-----------------|
| 인코더 | 상태·목표 **공유** φ | 상태 ψ, 목표 φ **분리** |
| 값 계산 | −‖φ(s) − φ(g)‖ (L2 거리) | ψ(s)^T φ(g) (내적) |
| φ 입력 | 상태 s | 목표 g만 |
| 목표 표현 | raw observation | 학습된 φ(g) |

---

## 3. 구현 계획

### 3.1 전체 구조

```
hilp_gcrl/
├── src/
│   ├── agents/
│   │   ├── hilp.py              (기존)
│   │   └── hilp_dual.py         [NEW] Dual Goal 에이전트
│   ├── special_networks.py      → DualGoalValue 추가
│   └── dataset_utils.py         → dual φ(g) 처리 추가
└── main.py                      → --agent_name=hilp_dual 분기 추가
```

### 3.2 Step 1: `DualGoalPhiValue` 네트워크 추가

파일: `hilp_gcrl/src/special_networks.py`

```python
class DualGoalPhiValue(nn.Module):
    """
    Dual Goal Representation value function.
    V(s, g) = psi(s)^T phi(g)
    psi: state encoder, phi: goal encoder (분리된 네트워크)
    """
    hidden_dims: tuple = (256, 256)
    skill_dim: int = 32          # φ 차원 (d)
    use_layer_norm: bool = True
    ensemble: bool = True
    encoder: nn.Module = None

    def setup(self):
        repr_class = LayerNormRepresentation if self.use_layer_norm else Representation
        # 상태 인코더 ψ(s) → R^skill_dim
        self.psi = repr_class((*self.hidden_dims, self.skill_dim),
                              activate_final=False, ensemble=self.ensemble)
        # 목표 인코더 φ(g) → R^skill_dim
        self.phi = repr_class((*self.hidden_dims, self.skill_dim),
                              activate_final=False, ensemble=self.ensemble)

    def get_phi(self, goals):
        """Dual goal representation: φ(g)"""
        return self.phi(goals)[0]  # 첫 번째 ensemble 멤버 반환

    def get_psi(self, observations):
        """State encoding: ψ(s)"""
        return self.psi(observations)[0]

    def __call__(self, observations, goals=None):
        psi_s = self.psi(observations)       # (B, skill_dim) or (2, B, skill_dim)
        phi_g = self.phi(goals)              # (B, skill_dim) or (2, B, skill_dim)
        # V(s,g) = ψ(s)^T φ(g), ensemble 각 멤버별 내적
        v = (psi_s * phi_g).sum(axis=-1)    # (B,) or (2, B)
        return v
```

> **주의**: ensemble=True일 때 shape이 (2, B, D)이므로 내적 계산 시 axis=-1로 처리.

### 3.3 Step 2: `hilp_dual.py` 에이전트 구현

파일: `hilp_gcrl/src/agents/hilp_dual.py`

**핵심 수정 사항**:

```python
# Phase 1: Temporal distance V(s,g) = ψ(s)^T φ(g)
def compute_value_loss(agent, batch, network_params):
    """기존 HILP expectile loss와 동일한 구조, 값 계산만 내적으로 변경"""
    # (변경 없음 - 네트워크 내부에서 내적 처리)
    ...

# Phase 2: φ(g)를 goal representation으로 사용
def loss_fn(network_params, agent, batch):
    # GCVF 손실 (Phase 1)
    value_loss, value_info = compute_value_loss(...)

    # batch['goals'] → φ(goals) 변환 (Phase 2)
    batch['goal_reps'] = agent.network(batch['goals'], method='phi_goal')

    # Skill policy: goal_reps를 goal 입력으로 사용
    batch['phis'] = agent.network(batch['observations'], method='psi_state')
    batch['next_phis'] = agent.network(batch['next_observations'], method='psi_state')
    # 스킬 보상: Δψ(s) 방향 (기존 HILP 방식 유지)
    batch['rewards'] = ((batch['next_phis'] - batch['phis']) * batch['skills']).sum(axis=1)
    ...
```

**HILPNetwork 확장**:

```python
class HILPDualNetwork(nn.Module):
    networks: Dict[str, nn.Module]

    def phi_goal(self, goals, **kwargs):
        """Phase 2: φ(g) 추출"""
        return self.networks['value'].get_phi(goals)

    def psi_state(self, observations, **kwargs):
        """ψ(s) 추출 - 스킬 보상 계산용"""
        return self.networks['value'].get_psi(observations)

    def value(self, observations, goals, **kwargs):
        return self.networks['value'](observations, goals, **kwargs)
    ...
```

### 3.4 Step 3: GCDataset 수정

파일: `hilp_gcrl/src/dataset_utils.py`

`sample()` 시 goal을 raw observation으로 유지하고, `loss_fn` 내부에서 φ(g) 변환.
(데이터셋 수정 최소화 원칙 — 배치 내에서 on-the-fly 변환)

### 3.5 Step 4: `main.py` 분기 추가

```python
if FLAGS.agent_name == 'hilp_dual':
    from src.agents import hilp_dual as learner
else:
    from src.agents import hilp as learner
```

실행:
```bash
python main.py \
  --agent_name=hilp_dual \
  --env_name=antmaze-large-diverse-v2 \
  --skill_dim=32
```

### 3.6 Step 5: 평가 시 φ(g) 사용

기존 `goal_skill` 평가 함수에서:
- `goal` (raw obs) → `phi_goal(goal)` 로 변환 후 정책에 입력
- planning도 φ-공간 KNN 대신 내적 거리 기반으로 수정

---

## 4. 하이퍼파라미터 권장값

| 파라미터 | 권장값 | 근거 |
|----------|--------|------|
| `skill_dim` | 32~64 | 논문 기본값 64 권장, 기존 HILP 32와 비교 실험 |
| `expectile` | 0.95 | 기존 HILP와 동일 |
| `discount` | 0.99 | 기존 HILP와 동일 |
| `lr` | 3e-4 | 기존 HILP와 동일 |
| `use_layer_norm` | True | 표현 안정성을 위해 필수 |

---

## 5. 실험 설계

### 5.1 Ablation
| 실험 | 설명 |
|------|------|
| HILP (baseline) | 기존 공유 인코더, L2 거리 |
| HILP-Dual (ours) | 분리 인코더, 내적 |
| HILP-Dual w/o Phase2 | φ(g) 미사용, raw goal 유지 |

### 5.2 환경
- `antmaze-umaze-v2` (쉬운 baseline)
- `antmaze-large-diverse-v2` (어려운 장거리)
- `kitchen-mixed-v0` (픽셀 기반은 선택)

### 5.3 기대 효과
논문 결과 기준:
- 20개 OGBench 태스크 전반에서 기존 GCRL 알고리즘 대비 일관된 성능 향상
- 특히 외생적 노이즈가 있는 환경에서 강건성 개선

---

## 6. 구현 우선순위

1. `DualGoalPhiValue` 네트워크 구현 (`special_networks.py`)
2. `HILPDualNetwork` wrapper 구현 (`special_networks.py`)
3. `hilp_dual.py` 에이전트 (loss_fn, create_learner)
4. `main.py` 분기 추가
5. 평가 함수 φ(g) 변환 적용
6. 실험 실행 및 비교

---

## 참고 문헌

- [Dual Goal Representations (arXiv:2510.06714)](https://arxiv.org/abs/2510.06714) - Seohong Park et al.
- [Foundation Policies with Hilbert Representations (arXiv:2402.15567)](https://arxiv.org/abs/2402.15567) - HILP 원논문
