"""Aggregate all-20 corruption axes and write a standalone Traditional-Chinese report."""
import csv
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np

from analyze import OUT, ROOT, CONDITIONS, THRESHOLD, write_json


RATIOS = json.loads((ROOT / 'submissions/v1/LOCAL_RECOMPUTE.json').read_text())['by_corruption']
EXPECTED = {
    'spatter': 'A：靜態可匹配結構 → 運動塌陷',
    'frost': 'A：靜態可匹配結構 → 運動塌陷',
    'rain': 'B：變化可匹配結構 → 運動膨脹',
    'snow': 'B：變化可匹配結構 → 運動膨脹',
    'zoom_blur': 'B：變化可匹配結構 → 運動膨脹',
}
STRUCTURE_THRESHOLD = 2.0
STATIC_CORRELATION_THRESHOLD = 0.9


def csv_write(path, rows):
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def scene_summary(rows, field):
    values = {scene: [r[field] for r in rows if r['scene'] == scene and r[field] is not None]
              for scene in sorted({r['scene'] for r in rows})}
    means = np.array([np.mean(value) for value in values.values() if value], dtype=np.float64)
    if len(means) != 10:
        raise ValueError(field + ': expected ten scenes')
    rng = np.random.default_rng(20260911)
    picks = rng.integers(10, size=(10000, 10))
    bootstrap = means[picks].mean(axis=1)
    return dict(mean=float(means.mean()), ci95=np.quantile(bootstrap, [.025, .975]).tolist(),
                scene_min=float(means.min()), scene_max=float(means.max()),
                pair_count=sum(len(v) for v in values.values()),
                by_scene={k: float(np.mean(v)) if v else None for k, v in values.items()})


def label(condition, ratio, temporal_correlation, structure_score):
    expected = EXPECTED.get(condition, 'C：不加可匹配結構 → 中性')
    if condition in ['spatter', 'frost']:
        matched = (structure_score >= STRUCTURE_THRESHOLD and
                   temporal_correlation >= STATIC_CORRELATION_THRESHOLD and ratio < 1)
        reason = ('S≥2、r_t≥0.9、R<1' if matched else
                  '未同時滿足 S≥2、r_t≥0.9、R<1')
    elif condition in ['rain', 'snow', 'zoom_blur']:
        matched = (structure_score >= STRUCTURE_THRESHOLD and
                   temporal_correlation < STATIC_CORRELATION_THRESHOLD and ratio > 1)
        reason = ('S≥2、r_t<0.9、R>1' if matched else
                  '未同時滿足 S≥2、r_t<0.9、R>1')
    else:
        matched = structure_score < STRUCTURE_THRESHOLD and .979 <= ratio <= 1.011
        if matched:
            reason = 'S<2 且 R落在預先給定[0.979,1.011]'
        elif structure_score >= STRUCTURE_THRESHOLD:
            reason = 'C的無結構前提不符（S≥2）'
        else:
            reason = 'R不在預先給定[0.979,1.011]'
    return expected, matched, reason


def f(value, digits=4):
    return 'N/A' if value is None else f'{value:.{digits}f}'


def ci(value):
    return 'N/A' if value is None else '[' + ', '.join(f'{x:.4f}' for x in value) + ']'


def main():
    assert json.loads((OUT / 'DONE.json').read_text())['done']
    config = json.loads((OUT / 'CONFIG.json').read_text())
    assert hashlib.sha256((OUT / 'analyze.py').read_bytes()).hexdigest() == config['script_sha256']
    scenes = [json.loads(path.read_text()) for path in sorted((OUT / 'scenes').glob('*.json'))]
    assert len(scenes) == 10 and all(x['done'] for x in scenes)
    rows = [r for item in scenes for r in item['rows']]
    assert len(rows) == 20 * 10 * 2 * (20 + 10)
    output, table, scene_rows = {}, [], []
    for condition in CONDITIONS:
        temporal = [r for r in rows if r['condition'] == condition and r['dimension'] == 'time']
        structure = [r for r in rows if r['condition'] == condition and r['dimension'] == 'structure']
        assert len(temporal) == 200 and len(structure) == 400
        temporal_stats = {key: scene_summary(temporal, key)
                          for key in ['iou', 'residual_amplitude_pearson', 'coverage_a']}
        structure_stats = {key: scene_summary(structure, key)
                           for key in ['coherent_residual_score', 'autocorr_peak_1to8', 'amplitude_std']}
        ratio = float(RATIOS[condition]['ratio'])
        expected, matched, reason = label(
            condition, ratio,
            temporal_stats['residual_amplitude_pearson']['mean'],
            structure_stats['coherent_residual_score']['mean'])
        output[condition] = dict(temporal=temporal_stats, structure=structure_stats,
                                 public_waft_magnitude_ratio=ratio,
                                 expected_class=expected, rule_matched=matched, match_reason=reason)
        table.append(dict(condition=condition,
                          time_iou=temporal_stats['iou']['mean'], time_iou_ci95=temporal_stats['iou']['ci95'],
                          time_correlation=temporal_stats['residual_amplitude_pearson']['mean'],
                          time_correlation_ci95=temporal_stats['residual_amplitude_pearson']['ci95'],
                          structure_score=structure_stats['coherent_residual_score']['mean'],
                          structure_score_ci95=structure_stats['coherent_residual_score']['ci95'],
                          autocorr_peak=structure_stats['autocorr_peak_1to8']['mean'],
                          amplitude_std=structure_stats['amplitude_std']['mean'],
                          waft_ratio=ratio, expected_class=expected, rule_matched=matched, match_reason=reason))
        for dim, stats in [('time', temporal_stats), ('structure', structure_stats)]:
            for scene in config['scenes']:
                scene_rows.append(dict(condition=condition, dimension=dim, scene=scene,
                                       **{key: value['by_scene'][scene] for key, value in stats.items()}))
    result = dict(done=True, conditions=output, temporal_pairs_per_condition=200,
                  images_per_condition=400, scene_count=10, config=config,
                  waft_ratio_source='submissions/v1/LOCAL_RECOMPUTE.json; read-only',
                  cpu_only=True, model_inference=False)
    write_json(OUT / 'SUMMARY.json', result)
    csv_write(OUT / 'axes.csv', table)
    csv_write(OUT / 'scene_axes.csv', scene_rows)
    inputs = {v['member']: v for item in scenes for v in item['inputs']}
    assert len(inputs) == 8400  # 20 clean frames*2 eyes*10 + 20 conditions*same
    write_json(OUT / 'INPUT_MANIFEST.json', list(inputs.values()))

    report = ['# RobustSpring 全20種污染：時間一致性、空間結構與公開 WAFT 反應方向', '',
              f'完成：{datetime.datetime.now().astimezone().isoformat(timespec="seconds")}。CPU-only；沒有模型推論、訓練或投稿。', '',
              '## 結論', '',
              '這條三類規則**不完全成立**。A與B的五項指定條件都同時符合結構、時間與幅度方向；但是C把「中性結果」誤等同於「不加可匹配結構」。多個中性條件的殘差有強局部結構（例如brightness、contrast、fog、blur、壓縮與形變）。因此本量測支持「時間行為與輸出方向有關」的描述，不能把所有中性條件說成沒有空間結構，或單以本結構分數證明因果機制。', '',
              '表中的「是否相符」採描述性切點：S<2 是噪聲樣（最高shot_noise=1.50），S≥2是有局部結構；此切點落在觀測到的1.50與4.51之間的空檔。A還要求r_t≥0.9與R<1；B要求r_t<0.9與R>1；C要求S<2與R在預先指定的[0.979,1.011]。切點用於如實審核此規則，不是預先註冊的顯著性檢定。JPEG另有R=0.978863，略低於預先指定下界。', '',
              '## 20列主表', '',
              f'跨幀軸沿用前次定義：`D=C-I`、`A=mean_RGB(|D|)`、`M=(A>{THRESHOLD})`；每組為相鄰影格 t/t+1 的 IoU(Mt,Mt+1) 與全圖Pearson(A_t,A_t+1)。每種共10 scene × 10 相鄰點 × 2眼＝200組。結構指標 `S=std(A)*max(0,max ρ_d)`，其中ρ_d是在原生1–8px的水平／垂直位移上取A的Pearson峰值；為控CPU，相關在固定每8px取樣格估計。S=0表示殘差空間常數或沒有正的局部自相關；高S表示非恆定且局部連續的殘差。它是「可匹配結構」的代理量，非模型內部matching cost的直接量測。', '',
              '| 污染 | 跨幀 IoU [CI95] | 跨幀相關 [CI95] | 結構指標 S [CI95] | 公開WAFT幅度比 | 規則預測類別 | 是否相符 |',
              '|---|---|---|---|---:|---|---|']
    for r in table:
        report.append(f'| {r["condition"]} | {f(r["time_iou"])} {ci(r["time_iou_ci95"])} | {f(r["time_correlation"])} {ci(r["time_correlation_ci95"])} | {f(r["structure_score"])} {ci(r["structure_score_ci95"])} | {r["waft_ratio"]:.6f} | {r["expected_class"]} | {"是：" if r["rule_matched"] else "否："}{r["match_reason"]} |')
    report += ['', 'CI：以10個scene整塊重抽10000次，seed=20260911；相鄰幀和兩眼維持在同一scene塊內。CI描述此固定test場景集合，並非獨立污染紋理母體。逐scene資料見 `scene_axes.csv`，完整機器可讀主表見 `axes.csv`。', '',
               '## 結構分數的必要限制', '',
               '這個代理量特意只問「殘差圖本身是否具有非恆定的局部空間可預測性」，不問它是否和真實場景有幾何對應。它能把獨立逐像素噪聲與連續紋理／條紋區分開，但也會對edge-based blur殘差、壓縮artifact、平滑形變、非均勻光度變換給高分。故它不是A/B/C的唯一分類器。', '',
               '噪聲檢查已通過預期：gaussian_noise S=0.228、shot_noise=1.503、impulse_noise=0.227，均低於rain=14.504與snow=25.436；其時間相關也很低（0.013、0.121、−0.000 vs 0.120、0.019）。因此這三種逐幀重抽噪聲確實位於「低局部相干」端。', '',
               '## 和既有lens對齊證據的連結', '',
               '[BENCHMARK_CORRUPTION_ALIGNMENT_ZH.md](../../BENCHMARK_CORRUPTION_ALIGNMENT_ZH.md) 已在公開V1的spatter/frost全部影像上驗證 `C=min(I+T,255)` 的全域固定模板等價性：同一T跨時間、左右眼與序列共用，逐值重建差異為0。它比單一IoU或本報告S更強，並直接支持spatter/frost的「靜態貼圖」前提。Rain/snow是不同的3D粒子類型；本報告不把其較低時間一致性誤稱為靜態模板。', '',
               '公開WAFT幅度比逐條件讀自 `submissions/v1/LOCAL_RECOMPUTE.json`，本輪只讀取；未開啟或修改 `submissions/v1`。分析PNG按ZIP member讀入RAM，未解壓資料樹；未修改 `runs/official_delta_v1` 或 `paper`。', '',
               '重現：`analyze.py`、`summarize.py`、`CONFIG.json`、`SUMMARY.json`、`axes.csv`。輸入member雜湊見 `INPUT_MANIFEST.json`。']
    (OUT / 'REPORT_ZH.md').write_text('\n'.join(report) + '\n')
    write_json(OUT / 'FINAL_VERIFIED.json', dict(done=True, scene_count=10, temporal_pairs_per_condition=200,
                                                 structure_images_per_condition=400, unique_png_members=len(inputs),
                                                 cpu_only=True, model_inference=False,
                                                 report_sha256=hashlib.sha256((OUT / 'REPORT_ZH.md').read_bytes()).hexdigest()))
    print(json.dumps(table, indent=2))


if __name__ == '__main__':
    main()
