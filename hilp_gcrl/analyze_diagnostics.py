"""
Analyze diagnostics.jsonl from train_dual_ogbench_diag.py.

Tests five hypotheses for value divergence:
  H1: psi/phi encoder norms grow unboundedly
  H2: V > 0 fraction grows (violates theoretical bound V ∈ [-200, 0])
  H3: target_V bootstrapping creates positive feedback loop
  H4: Cosine similarity concentration (psi and phi align in a narrow cone)
  H5: inline Q (no separate Q network) amplifies bootstrapping instability

Usage:
  python3 analyze_diagnostics.py <path/to/diagnostics.jsonl>
"""

import json
import sys
import os

import numpy as np


def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def print_section(title):
    print(f'\n{"=" * 70}')
    print(f'  {title}')
    print(f'{"=" * 70}')


def analyze_h1_encoder_norms(records):
    """H1: Do psi/phi norms grow unboundedly?"""
    print_section('H1: Encoder Norm Growth (psi & phi)')

    steps = [r['step'] for r in records]
    psi_mean = [r.get('diag/psi_norm_mean', 0) for r in records]
    psi_max  = [r.get('diag/psi_norm_max', 0) for r in records]
    phi_mean = [r.get('diag/phi_norm_mean', 0) for r in records]
    phi_max  = [r.get('diag/phi_norm_max', 0) for r in records]

    # Print first and last 5 records
    print(f'  {"Step":>8s}  {"psi_mean":>10s}  {"psi_max":>10s}  {"phi_mean":>10s}  {"phi_max":>10s}')
    print(f'  {"-"*8}  {"-"*10}  {"-"*10}  {"-"*10}  {"-"*10}')
    for i in list(range(min(5, len(records)))) + list(range(max(5, len(records)-5), len(records))):
        if i >= len(records):
            continue
        print(f'  {steps[i]:>8d}  {psi_mean[i]:>10.4f}  {psi_max[i]:>10.4f}  '
              f'{phi_mean[i]:>10.4f}  {phi_max[i]:>10.4f}')

    # Growth ratio
    if len(records) >= 2 and psi_mean[0] > 0 and phi_mean[0] > 0:
        psi_growth = psi_mean[-1] / psi_mean[0]
        phi_growth = phi_mean[-1] / phi_mean[0]
        print(f'\n  psi norm growth ratio (last/first): {psi_growth:.2f}x')
        print(f'  phi norm growth ratio (last/first): {phi_growth:.2f}x')

        if psi_growth > 5 or phi_growth > 5:
            print(f'  >>> CONFIRMED: Encoder norms are growing significantly.')
            print(f'      Inner product V = psi^T phi grows as product of norms.')
        else:
            print(f'  >>> Encoder norms are relatively stable.')

    # Check if growth is exponential
    if len(psi_mean) >= 10:
        first_half_max = max(psi_mean[:len(psi_mean)//2])
        second_half_max = max(psi_mean[len(psi_mean)//2:])
        if first_half_max > 0:
            accel = second_half_max / first_half_max
            print(f'  psi norm acceleration (2nd half max / 1st half max): {accel:.2f}x')


def analyze_h2_v_positive(records):
    """H2: Does V > 0 fraction grow? (Should be 0 at convergence)"""
    print_section('H2: V > 0 Violation (theoretical bound: V <= 0)')

    steps = [r['step'] for r in records]
    v_pos  = [r.get('diag/v_pos_frac', 0) for r in records]
    v_p50  = [r.get('diag/v_online_p50', 0) for r in records]
    v_p95  = [r.get('diag/v_online_p95', 0) for r in records]
    v_p99  = [r.get('diag/v_online_p99', 0) for r in records]
    v_max  = [r.get('value/v_max', 0) for r in records]

    print(f'  {"Step":>8s}  {"V>0 frac":>10s}  {"V_p50":>10s}  {"V_p95":>10s}  '
          f'{"V_p99":>10s}  {"V_max":>10s}')
    print(f'  {"-"*8}  {"-"*10}  {"-"*10}  {"-"*10}  {"-"*10}  {"-"*10}')
    for i in list(range(min(5, len(records)))) + list(range(max(5, len(records)-5), len(records))):
        if i >= len(records):
            continue
        print(f'  {steps[i]:>8d}  {v_pos[i]:>10.4f}  {v_p50[i]:>10.2f}  '
              f'{v_p95[i]:>10.2f}  {v_p99[i]:>10.2f}  {v_max[i]:>10.2f}')

    # Is it the tail or the bulk?
    if len(records) > 5:
        late = records[-5:]
        avg_pos_frac = np.mean([r.get('diag/v_pos_frac', 0) for r in late])
        avg_p50 = np.mean([r.get('diag/v_online_p50', 0) for r in late])
        avg_p99 = np.mean([r.get('diag/v_online_p99', 0) for r in late])
        avg_vmax = np.mean([r.get('value/v_max', 0) for r in late])

        print(f'\n  Late-stage (last 5 records):')
        print(f'    avg V>0 fraction: {avg_pos_frac:.4f}')
        print(f'    avg V_p50: {avg_p50:.2f}  avg V_p99: {avg_p99:.2f}  avg V_max: {avg_vmax:.2f}')

        if avg_p50 > 0:
            print(f'  >>> SEVERE: Even the MEDIAN V is positive. Bulk divergence.')
        elif avg_pos_frac > 0.1:
            print(f'  >>> MODERATE: >10% of V values are positive. Growing tail problem.')
        elif avg_vmax > 100:
            print(f'  >>> MILD: V_max is large but bulk is still negative. Outlier-driven.')


def analyze_h3_bootstrap_feedback(records):
    """H3: Does target_V track online_V too closely (positive feedback)?"""
    print_section('H3: Target V Bootstrapping Feedback Loop')

    steps = [r['step'] for r in records]
    tv_max   = [r.get('diag/target_v_max', 0) for r in records]
    v_max    = [r.get('value/v_max', 0) for r in records]
    iq_max   = [r.get('diag/inline_q_max', 0) for r in records]
    gap_max  = [r.get('diag/v_target_gap_max', 0) for r in records]
    tnv_max  = [r.get('diag/target_next_v_max', 0) for r in records]

    print(f'  {"Step":>8s}  {"V_max":>10s}  {"tgtV_max":>10s}  {"inQ_max":>12s}  '
          f'{"tgtNxV_max":>12s}  {"gap_max":>10s}')
    print(f'  {"-"*8}  {"-"*10}  {"-"*10}  {"-"*12}  {"-"*12}  {"-"*10}')
    for i in list(range(min(5, len(records)))) + list(range(max(5, len(records)-5), len(records))):
        if i >= len(records):
            continue
        print(f'  {steps[i]:>8d}  {v_max[i]:>10.2f}  {tv_max[i]:>10.2f}  '
              f'{iq_max[i]:>12.2f}  {tnv_max[i]:>12.2f}  {gap_max[i]:>10.2f}')

    # Check correlation between v_max and target_v_max
    if len(records) >= 5:
        corr = np.corrcoef(v_max, tv_max)[0, 1]
        print(f'\n  Correlation(V_max, target_V_max): {corr:.4f}')
        if corr > 0.9:
            print(f'  >>> CONFIRMED: target_V closely tracks online V. Positive feedback loop.')
            print(f'      Without a separate Q network, V directly bootstraps from EMA(V).')
            print(f'      This is the paper vs code structural difference.')

    # Check if inline Q exceeds theoretical maximum
    if len(records) > 0:
        max_iq = max(iq_max)
        print(f'\n  Max inline Q seen: {max_iq:.2f}')
        print(f'  Theoretical Q max: 0 (since r ∈ [-1, 0] and V* ∈ [-200, 0])')
        if max_iq > 10:
            print(f'  >>> inline Q is wildly above theoretical bound.')


def analyze_h4_cosine_similarity(records):
    """H4: Are psi and phi aligning in a narrow cone?"""
    print_section('H4: Cosine Similarity (psi-phi alignment)')

    steps = [r['step'] for r in records]
    cos_mean = [r.get('diag/cos_sim_mean', 0) for r in records]
    cos_max  = [r.get('diag/cos_sim_max', 0) for r in records]
    cos_min  = [r.get('diag/cos_sim_min', 0) for r in records]

    print(f'  {"Step":>8s}  {"cos_mean":>10s}  {"cos_max":>10s}  {"cos_min":>10s}')
    print(f'  {"-"*8}  {"-"*10}  {"-"*10}  {"-"*10}')
    for i in list(range(min(5, len(records)))) + list(range(max(5, len(records)-5), len(records))):
        if i >= len(records):
            continue
        print(f'  {steps[i]:>8d}  {cos_mean[i]:>10.4f}  {cos_max[i]:>10.4f}  {cos_min[i]:>10.4f}')

    if len(records) > 5:
        late_cos_mean = np.mean([r.get('diag/cos_sim_mean', 0) for r in records[-5:]])
        early_cos_mean = np.mean([r.get('diag/cos_sim_mean', 0) for r in records[:5]])
        print(f'\n  Early avg cosine sim: {early_cos_mean:.4f}')
        print(f'  Late  avg cosine sim: {late_cos_mean:.4f}')
        if late_cos_mean > 0.5:
            print(f'  >>> psi and phi are aligning. Inner product grows even without norm growth.')


def analyze_h5_ensemble_stability(records):
    """H5: Is the ensemble spread stable?"""
    print_section('H5: Ensemble & Inline Q Stability')

    steps = [r['step'] for r in records]
    v1_mean = [r.get('diag/v1_mean', 0) for r in records]
    v2_mean = [r.get('diag/v2_mean', 0) for r in records]
    gap     = [r.get('diag/v_ensemble_gap', 0) for r in records]
    delta_psi = [r.get('diag/delta_psi_mean', 0) for r in records]

    print(f'  {"Step":>8s}  {"V1_mean":>10s}  {"V2_mean":>10s}  {"ens_gap":>10s}  {"delta_psi":>10s}')
    print(f'  {"-"*8}  {"-"*10}  {"-"*10}  {"-"*10}  {"-"*10}')
    for i in list(range(min(5, len(records)))) + list(range(max(5, len(records)-5), len(records))):
        if i >= len(records):
            continue
        print(f'  {steps[i]:>8d}  {v1_mean[i]:>10.4f}  {v2_mean[i]:>10.4f}  '
              f'{gap[i]:>10.4f}  {delta_psi[i]:>10.4f}')


def summary_and_recommendations(records):
    """Print overall diagnosis and recommendations."""
    print_section('SUMMARY & RECOMMENDATIONS')

    if len(records) < 3:
        print('  Not enough data points for analysis.')
        return

    # Collect key indicators
    psi_growth = records[-1].get('diag/psi_norm_mean', 1) / max(records[0].get('diag/psi_norm_mean', 1), 1e-8)
    phi_growth = records[-1].get('diag/phi_norm_mean', 1) / max(records[0].get('diag/phi_norm_mean', 1), 1e-8)
    late_v_max = np.mean([r.get('value/v_max', 0) for r in records[-3:]])
    late_v_pos_frac = np.mean([r.get('diag/v_pos_frac', 0) for r in records[-3:]])
    late_grad_norm = np.mean([r.get('grad/norm', 0) for r in records[-3:]])
    late_loss = np.mean([r.get('loss', 0) for r in records[-3:]])

    print(f'  Key late-stage indicators:')
    print(f'    V_max:       {late_v_max:.2f}  (should be <= 0)')
    print(f'    V>0 frac:    {late_v_pos_frac:.4f}  (should be ~0)')
    print(f'    grad norm:   {late_grad_norm:.2f}')
    print(f'    loss:        {late_loss:.2f}')
    print(f'    psi growth:  {psi_growth:.2f}x')
    print(f'    phi growth:  {phi_growth:.2f}x')

    print(f'\n  Root cause analysis:')

    issues = []

    if psi_growth > 3 or phi_growth > 3:
        issues.append(('ENCODER NORM GROWTH',
            'psi/phi norms are growing, causing inner product V = psi^T phi to explode.\n'
            '    Fix: Add L2 normalization to psi and phi outputs, or add weight decay.'))

    if late_v_max > 50:
        issues.append(('UNBOUNDED INNER PRODUCT',
            'V(s,g) = psi(s)^T phi(g) has no upper bound. True V* in [-200, 0].\n'
            '    Fix: Clamp V to [-1/(1-gamma), 0] or add gradient clipping.'))

    tv_corr_data = [r.get('diag/target_v_max', 0) for r in records]
    v_max_data = [r.get('value/v_max', 0) for r in records]
    if len(tv_corr_data) >= 5:
        corr = np.corrcoef(v_max_data, tv_corr_data)[0, 1]
        if corr > 0.85:
            issues.append(('BOOTSTRAP FEEDBACK LOOP',
                f'target_V closely tracks online V (corr={corr:.3f}).\n'
                '    The paper uses a SEPARATE Q network (Algorithm 1, Eq. 3-4).\n'
                '    Current code computes Q = r + gamma*target_V inline.\n'
                '    Fix: Implement separate Q(s,a,g) network as in the paper.'))

    if late_grad_norm > 1000:
        issues.append(('GRADIENT EXPLOSION',
            f'grad norm = {late_grad_norm:.0f}. No gradient clipping is applied.\n'
            '    Fix: Add optax.clip_by_global_norm(max_norm=1.0) to optimizer.'))

    if not issues:
        print('    No clear issues detected in the available data.')
    else:
        for i, (name, desc) in enumerate(issues, 1):
            print(f'\n  [{i}] {name}')
            print(f'    {desc}')

    print(f'\n  Recommended fixes (in order of priority):')
    print(f'    1. Add gradient clipping: optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr))')
    print(f'    2. Add separate Q network as in paper Algorithm 1 (Eq. 3-4)')
    print(f'    3. Fix batch_size: 2048 → 1024 (paper default)')
    print(f'    4. Consider clamping V or normalizing psi/phi outputs')


def main():
    if len(sys.argv) < 2:
        print(f'Usage: python3 {sys.argv[0]} <diagnostics.jsonl>')
        sys.exit(1)

    path = sys.argv[1]
    if not os.path.exists(path):
        print(f'File not found: {path}')
        sys.exit(1)

    records = load_jsonl(path)
    print(f'Loaded {len(records)} diagnostic records from {path}')
    print(f'Steps: {records[0]["step"]} → {records[-1]["step"]}')

    analyze_h1_encoder_norms(records)
    analyze_h2_v_positive(records)
    analyze_h3_bootstrap_feedback(records)
    analyze_h4_cosine_similarity(records)
    analyze_h5_ensemble_stability(records)
    summary_and_recommendations(records)


if __name__ == '__main__':
    main()
