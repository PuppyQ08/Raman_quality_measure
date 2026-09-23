"""Project complete comparison profiles and illustrative intervals; no new tests."""
from pathlib import Path
import csv
import hashlib
import json

HERE = Path(__file__).resolve().parent
source = HERE / 'data/table_s1_full_alignment.csv'
with source.open() as stream:
    rows = list(csv.DictReader(stream))
lookup = {(r['panel_id'], r['metric_output_id']): r for r in rows}
assert len(rows) == len(lookup) == 143
panels = ['d1_a', 'd2_5_a', 'd2_5_b', 'd2_10_a', 'd2_10_b',
          'd2_20_a', 'd2_20_b', 'd4_a', 'd4_b', 'd5_a', 'd5_b']
metrics = ['rmse', 'mae', 'sam', 'pearson_r', 'nmse', 'wasserstein_1_cm1',
           'is_like_structure_to_noise', 'precision', 'recall', 'f1',
           'artifact_peak_ratio', 'missing_peak_ratio']

def decision(row, key):
    value = float(row[key])
    if float(row[key + '_adjusted_p_value']) >= .05 or value == 0:
        return 'not_rejected'
    return 'favorable' if value > 0 else 'adverse'

profiles = []
for panel in panels:
    for metric in metrics:
        r = lookup[panel, metric]
        ag, acc = decision(r, 'd_ag'), decision(r, 'd_acc')
        joint = (ag if ag == acc and ag != 'not_rejected' else
                 'tradeoff' if {ag, acc} == {'favorable', 'adverse'} else 'not_joint')
        profiles.append(dict(panel_id=panel, metric_output_id=metric,
                             ag_decision=ag, concordance_decision=acc,
                             joint_decision=joint))

with (HERE / 'data/metric_profiles_v2.csv').open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(profiles[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(profiles)

structural = metrics[6:]
for metric in structural:
    rr = [r for r in profiles if r['metric_output_id'] == metric]
    assert [r['panel_id'] for r in rr if r['joint_decision'] == 'favorable'] == ['d4_b']
    assert sum(r['joint_decision'] == 'adverse' for r in rr) == 10
w1_good = [r['panel_id'] for r in profiles
           if r['metric_output_id'] == 'wasserstein_1_cm1' and r['joint_decision'] == 'favorable']
assert w1_good == ['d1_a', 'd2_5_a', 'd2_10_a', 'd2_20_a']
assert not any(r['joint_decision'] == 'favorable' for r in profiles
               if r['panel_id'].startswith('d5_'))
assert not any(r['ag_decision'] == 'favorable' or r['concordance_decision'] == 'favorable'
               for r in profiles if r['metric_output_id'] == 'rmse')
assert not any(r['joint_decision'] == 'tradeoff' for r in profiles)

examples = [
    ('d2_10_a', 'wasserstein_1_cm1', 'Bacteria / A', '$W_1$'),
    ('d2_10_a', 'mae', 'Bacteria / A', 'MAE'),
    ('d2_10_a', 'nmse', 'Bacteria / A', 'NMSE'),
    ('d2_10_b', 'wasserstein_1_cm1', 'Bacteria / B', '$W_1$'),
    ('d4_a', 'pearson_r', 'Sugar / A', 'Pearson'),
    ('d4_b', 'f1', 'Sugar / B', 'Peak F1'),
    ('d4_b', 'is_like_structure_to_noise', 'Sugar / B', 'S/N'),
    ('d4_b', 'wasserstein_1_cm1', 'Sugar / B', '$W_1$'),
    ('d4_b', 'nmse', 'Sugar / B', 'NMSE'),
    ('d5_a', 'wasserstein_1_cm1', 'Minerals / A', '$W_1$'),
    ('d5_a', 'mae', 'Minerals / A', 'MAE'),
    ('d5_b', 'wasserstein_1_cm1', 'Minerals / B', '$W_1$'),
]

def fmt(value):
    v = float(value)
    return f'{0 if abs(v) < .00005 else v:.4f}'

def interval(r, key):
    return f"{fmt(r[key])} [{fmt(r[key+'_lower'])}, {fmt(r[key+'_upper'])}]"

def pvalue(value):
    return '$<0.001$' if float(value) < .001 else f'{float(value):.3f}'

with (HERE / 'tables/representative_contrasts.tex').open('w') as f:
    for panel, metric, label, name in examples:
        r = lookup[panel, metric]
        f.write(' & '.join([label, name, interval(r, 'd_ag'), pvalue(r['d_ag_adjusted_p_value']),
                            interval(r, 'd_acc'), pvalue(r['d_acc_adjusted_p_value'])]) + r' \\' + '\n')

report = {
    'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
    'source_rows': len(rows), 'metric_outputs': 13, 'task_protocol_combinations': len(panels),
    'candidate_profiles': len(profiles), 'representative_rows': len(examples),
    'joint_favorable_w1_panels': w1_good,
    'structural_outputs_favorable_only_in_sugar_b_and_adverse_in_other_ten': structural,
    'no_joint_favorable_mineral_candidate': True,
    'significant_opposite_direction_tradeoffs': 0,
    'new_scientific_tests': False,
    'classification_rule': 'Contrast sign and original two-sided Holm-adjusted p < 0.05 for each diagnostic',
}
(HERE / 'provenance/v2_profile_validation.json').write_text(json.dumps(report, indent=2) + '\n')
print(f'Verified 132 candidate profiles; wrote {len(examples)} illustrative contrast intervals with original p-values.')
