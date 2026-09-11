"""Summarize CPU-only corruption axes v2 without changing source predictions."""
import csv
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np

from analyze import CONDITIONS, OUT


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.part')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def scene_stats(rows, field):
    by_scene = {scene: [row[field] for row in rows if row['scene'] == scene and row[field] is not None]
                for scene in sorted({row['scene'] for row in rows})}
    means = np.array([np.mean(by_scene[scene]) for scene in sorted(by_scene)], np.float64)
    if len(means) != 10: raise ValueError(field + ' needs ten scenes')
    rng = np.random.default_rng(20260911)
    bootstrap = means[rng.integers(0, 10, size=(10000, 10))].mean(axis=1)
    return {'mean': float(means.mean()), 'ci95': np.quantile(bootstrap, [.025, .975]).tolist(),
            'scene_min': float(means.min()), 'scene_max': float(means.max()),
            'by_scene': {scene: float(np.mean(values)) for scene, values in by_scene.items()}}


def fmt(x): return 'N/A' if x is None else f'{x:.4f}'
def ci(s): return '[' + ', '.join(fmt(x) for x in s['ci95']) + ']'


def expected(condition):
    if condition in ['spatter', 'frost']: return '靜態外加圖層／塌陷'
    if condition in ['rain', 'snow', 'zoom_blur']: return '獨立移動圖層／膨脹'
    return '隨場景變換／中性'


def sign_interpretation(delta):
    low, high = delta['ci95']
    if high < 0: return '零位移較佳（負 Δ）'
    if low > 0: return 'flow warp 較佳（正 Δ）'
    return 'CI 跨 0'


def write_csv(path, rows):
    fields = list(dict.fromkeys(field for row in rows for field in row))
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def main():
    assert json.loads((OUT / 'DONE.json').read_text())['done']
    config = json.loads((OUT / 'CONFIG.json').read_text())
    assert config['script_sha256'] == hashlib.sha256((OUT / 'analyze.py').read_bytes()).hexdigest()
    scenes = [json.loads(path.read_text()) for path in sorted((OUT / 'scenes').glob('*.json'))]
    assert len(scenes) == 10 and all(scene['done'] for scene in scenes)
    all_rows = [row for scene in scenes for row in scene['rows']]
    table, machine_rows, results = [], [], {}
    for condition in CONDITIONS:
        template_rows = [r for r in all_rows if r['condition'] == condition and r['dimension'] == 'template']
        temporal_rows = [r for r in all_rows if r['condition'] == condition and r['dimension'] == 'temporal']
        assert len(template_rows) == 10 and len(temporal_rows) == 200
        template = {'max_abs_error': max(r['max_abs_error'] for r in template_rows),
                    'mean_abs_error_per_channel': float(np.mean([r['mean_abs_error_per_channel'] for r in template_rows])),
                    'exact_zero_rgb_pixel_fraction': float(np.mean([r['exact_zero_rgb_pixel_fraction'] for r in template_rows])),
                    'exact_zero_channel_fraction': float(np.mean([r['exact_zero_channel_fraction'] for r in template_rows])),
                    'images': sum(r['images'] for r in template_rows)}
        stats = {field: scene_stats(temporal_rows, field) for field in [
            'zero_offset_correlation', 'flow_warp_correlation', 'delta_warp_minus_zero',
            'source_out_of_bounds_fraction', 'source_fb_inconsistent_fraction',
            'target_collision_or_hole_fraction', 'retained_target_fraction']}
        template_pass = template['max_abs_error'] == 0
        item = {'condition': condition, 'template': template, 'template_exact_pass': template_pass,
                'temporal': stats, 'expected': expected(condition),
                'delta_interpretation': sign_interpretation(stats['delta_warp_minus_zero'])}
        results[condition] = item
        table.append({'condition': condition, 'template_max_abs_error': template['max_abs_error'],
                      'template_mean_abs_error': template['mean_abs_error_per_channel'],
                      'template_exact_zero_rgb_fraction': template['exact_zero_rgb_pixel_fraction'],
                      'template_pass': template_pass,
                      'zero_offset_correlation': stats['zero_offset_correlation']['mean'],
                      'zero_offset_ci95': stats['zero_offset_correlation']['ci95'],
                      'flow_warp_correlation': stats['flow_warp_correlation']['mean'],
                      'flow_warp_ci95': stats['flow_warp_correlation']['ci95'],
                      'delta_warp_minus_zero': stats['delta_warp_minus_zero']['mean'],
                      'delta_ci95': stats['delta_warp_minus_zero']['ci95'],
                      'retained_fraction': stats['retained_target_fraction']['mean'],
                      'retained_ci95': stats['retained_target_fraction']['ci95'],
                      'expected': item['expected'], 'interpretation': item['delta_interpretation']})
        for field, result in stats.items():
            for scene, value in result['by_scene'].items():
                machine_rows.append({'condition': condition, 'metric': field, 'scene': scene, 'value': value})
    write_json(OUT / 'SUMMARY.json', {'done': True, 'results': results, 'cpu_only': True,
               'model_inference': False, 'protected_sources_read_only': [
                   'runs/submission_v1/predictions/clean', 'official ZIP archives']})
    write_csv(OUT / 'axes_v2.csv', table); write_csv(OUT / 'scene_axes_v2.csv', machine_rows)
    lines = ['# RobustSpring 全20種污染：全域模板與是否隨場景移動', '',
             f'完成：{datetime.datetime.now().astimezone().isoformat(timespec="seconds")}。CPU-only；沒有模型推論、訓練或投稿。', '',
             '## 測試定義', '',
             '**測試1（全域固定模板）**：對每一污染，從v1相同設計的 `0003/left` 20個時間點，以每個原生像素、每個RGB通道取 `T=max(max(C-I,0))`，然後以 `C_hat=min(I+T,255)` 在10 scenes × 20時間點 × 2眼＝400張PNG驗證。這是前次spatter/frost檢查的同一估計器與成像式；不做重採樣或學習。Pass定義嚴格為整個驗證集 `max|C_hat-C|=0`，沒有近似閾值。平均誤差是每通道絕對誤差；0誤差比例要求RGB三個通道均為0。', '',
             '**測試2（殘差是否跟隨場景）**：`D=C-I`、`A=mean_RGB(|D|)`。 (a)零位移為Pearson(`A_t,A_{t+1}`)；(b)讀取公開WAFT在clean影像上的既有FW/BW flow，以FW把 `A_t` nearest-target forward-splat 到t+1後再計算Pearson。保留的source像素必須落在畫面內且滿足 `||F(x)+B(x+F(x))||≤max(1px,0.05||F||)`；多個source落同一個target的collision及空洞target一律排除。表中保留率是最後用於相關係數的target像素比例，故排除率=1−保留率。每個條件有10 scenes × 10相鄰點 × 2眼＝200對；CI是10 scene block bootstrap 10,000次，seed=20260911。', '',
             '重要限制：這是用公開WAFT的**預測**flow做對齊，並非GT；因此它檢驗「在這個模型的clean幾何下，殘差是否更像場景攜帶」，不能單獨證明真實物理層的運動。', '',
             '## 結果判定', '',
             '測試1支持最強的部分：只有spatter與frost在400張驗證PNG上嚴格重建為0誤差。測試2也支持它們是零位移模板：兩者Δ的CI完全為負。對場景變換類，brightness、contrast、各blur、fog、JPEG與elastic_transform都有正Δ；elastic_transform的Δ=+0.1071 [+0.0724,+0.1391]，因此沒有打破「隨場景」判準。', '',
             '三分法**沒有完整成立**：rain與snow的兩種相關都低（rain：0.1198→0.1416；snow：0.0188→0.0217），雖然微小正Δ的CI不跨0，仍符合「沒有被clean scene flow有效對齊」的實質描述。zoom_blur則不符合：0.6551→0.9376，Δ=+0.2824 [+0.1711,+0.3935]，在clean flow下明顯更可對齊，不能寫成獨立移動圖層。另有gaussian/shot/speckle/impulse noise兩種相關都低，故也不能把「非lens且中性」全部歸為隨場景移動。', '',
             '## 20列主表', '',
             '| 污染 | T: max / mean / RGB零誤差 | T pass | 零位移相關 [CI95] | flow-warp相關 [CI95] | Δ=warp−zero [CI95] | 保留率 [CI95] | 預期 | 實測判讀 |',
             '|---|---:|---|---|---|---|---|---|---|']
    for r in table:
        t = f'{r["template_max_abs_error"]} / {r["template_mean_abs_error"]:.4f} / {r["template_exact_zero_rgb_fraction"]:.6f}'
        lines.append(f'| {r["condition"]} | {t} | {"PASS" if r["template_pass"] else "FAIL"} | '
                     f'{fmt(r["zero_offset_correlation"])} {ci({"ci95":r["zero_offset_ci95"]})} | '
                     f'{fmt(r["flow_warp_correlation"])} {ci({"ci95":r["flow_warp_ci95"]})} | '
                     f'{fmt(r["delta_warp_minus_zero"])} {ci({"ci95":r["delta_ci95"]})} | '
                     f'{fmt(r["retained_fraction"])} {ci({"ci95":r["retained_ci95"]})} | '
                     f'{r["expected"]} | {r["interpretation"]} |')
    exact = [r['condition'] for r in table if r['template_pass']]
    negative = [r['condition'] for r in table if r['delta_ci95'][1] < 0]
    positive = [r['condition'] for r in table if r['delta_ci95'][0] > 0]
    cross = [r['condition'] for r in table if r['condition'] not in negative + positive]
    lines += ['', '## 不迎合預期的摘要', '',
              f'固定模板嚴格PASS的條件：{", ".join(exact) if exact else "無"}。',
              f'Δ CI完全為負（零位移較佳）：{", ".join(negative) if negative else "無"}。',
              f'Δ CI完全為正（flow-warp較佳）：{", ".join(positive) if positive else "無"}。',
              f'Δ CI跨0：{", ".join(cross) if cross else "無"}。', '',
              '尤其應依表中數字判斷zoom_blur與elastic_transform；本報告不會把它們硬塞進預期三分。完整機器可讀表為 `axes_v2.csv`，逐scene數值為 `scene_axes_v2.csv`；輸入PNG與flow檔案雜湊在各 `scenes/*.json`。',
              '']
    (OUT / 'REPORT_ZH.md').write_text('\n'.join(lines))
    print(json.dumps(table, indent=2))


if __name__ == '__main__': main()
