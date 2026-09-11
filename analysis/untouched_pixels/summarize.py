"""Aggregate frozen public-WAFT dense predictions by native RGB-identical subsets."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENBLAS_NUM_THREADS']='1'
import csv
import datetime
import hashlib
import json
from pathlib import Path
import numpy as np
from analyze import OUT,PRED,CONFIG_SHA,CONDITIONS,H,W,write_json


def merge(rows):
    return {k:sum(r[k] for r in rows) for k in ['n','epe_sum','clean_mag_sum','corrupt_mag_sum']}


def finalize(s,total):
    n=s['n']
    return dict(pixel_observations=n,fraction=n/total,
                epe=s['epe_sum']/n if n else None,
                clean_mean_magnitude=s['clean_mag_sum']/n if n else None,
                corrupt_mean_magnitude=s['corrupt_mag_sum']/n if n else None,
                magnitude_ratio=s['corrupt_mag_sum']/s['clean_mag_sum'] if n and s['clean_mag_sum']>0 else None)


def bootstrap(scene_sums,scene_totals):
    values=list(scene_sums.values());nscene=len(values)
    assert nscene==10
    rng=np.random.default_rng(20260910);picks=rng.integers(nscene,size=(10000,nscene))
    arr={k:np.array([v[k] for v in values],dtype=np.float64)[picks].sum(1) for k in values[0]}
    total=np.array(list(scene_totals.values()),dtype=np.float64)[picks].sum(1)
    output={}
    for field,numerator,denominator in [('epe',arr['epe_sum'],arr['n']),
                                         ('magnitude_ratio',arr['corrupt_mag_sum'],arr['clean_mag_sum']),
                                         ('fraction',arr['n'],total)]:
        valid=denominator>0
        support=sum(v['n']>0 for v in values) if field=='epe' else sum(v['clean_mag_sum']>0 for v in values) if field=='magnitude_ratio' else nscene
        output[field+'_ci95']=np.quantile(numerator[valid]/denominator[valid],[.025,.975]).tolist() if valid.any() and support>=2 else None
        output[field+'_ci95_status']='conditional on valid bootstrap draws' if valid.any() and support>=2 else 'fewer than two supporting scenes; interval not estimable'
        output[field+'_bootstrap_valid_draws']=int(valid.sum())
    return output


def csv_write(path,rows):
    with path.open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def number(value,digits=4):return 'N/A' if value is None else f'{value:.{digits}f}'


def interval(value,digits=4):return 'N/A' if value is None else '['+', '.join(f'{v:.{digits}f}' for v in value)+']'


def main():
    assert json.loads((OUT/'DONE.json').read_text())['done']
    cfg=json.loads((OUT/'CONFIG.json').read_text())
    assert hashlib.sha256((OUT/'analyze.py').read_bytes()).hexdigest()==cfg['script_sha256']
    assert hashlib.sha256((PRED/'inference_config.json').read_bytes()).hexdigest()==CONFIG_SHA
    data=[json.loads(p.read_text()) for p in sorted((OUT/'workers').glob('*.json'))]
    assert len(data)==20 and all(d['done'] and d['config_sha256']==CONFIG_SHA for d in data)
    rows=[r for d in data for r in d['rows']]
    keys={(r['scene'],r['camera'],r['direction'],r['source'],r['condition']) for r in rows}
    assert len(rows)==len(keys)==19800
    scopes=list(rows[0]['stats'])
    pred_files={p['path']:p for d in data for p in d['prediction_files']}
    assert len(pred_files)==23760
    inputs={p['member']:p for d in data for p in d['inputs']}
    assert len(inputs)==12000
    scene_names=sorted(cfg['scenes'])
    # Source RGB masks are direction independent. Deduplicate FW/BW observations to
    # obtain the requested unique-image untouched proportion as an additional statistic.
    image_counts={}
    for r in rows:
        key=(r['condition'],r['scene'],r['camera'],r['source'])
        count=r['stats']['source_untouched']['n']
        assert key not in image_counts or image_counts[key]==count
        image_counts[key]=count
        for kind in ['source','both']:
            u=r['stats'][kind+'_untouched'];t=r['stats'][kind+'_touched'];a=r['stats']['all']
            assert u['n']+t['n']==a['n']==H*W
            for k in ['epe_sum','clean_mag_sum','corrupt_mag_sum']:
                assert np.isclose(u[k]+t[k],a[k],rtol=1e-12,atol=1e-7)
        assert r['stats']['local5_untouched']['n']<=r['stats']['both_untouched']['n']
        assert r['stats']['correspondence_local5_untouched']['n']<=r['stats']['both_untouched']['n']
    assert len(image_counts)==10000
    summaries={};table=[];pair_table=[];scene_table=[]
    for c in CONDITIONS:
        cr=[r for r in rows if r['condition']==c];assert len(cr)==3960
        total=3960*H*W
        img=[v for (cond,*_),v in image_counts.items() if cond==c]
        assert len(img)==2000
        summaries[c]=dict(directed_flow_pairs=3960,total_pixel_observations=total,
                          unique_images=2000,unique_image_rgb_identical_pixels=sum(img),
                          unique_image_rgb_identical_fraction=sum(img)/(2000*H*W),
                          unique_images_with_untouched_pixels=sum(v>0 for v in img),subsets={})
        scene_totals={scene:sum(r['stats']['all']['n'] for r in cr if r['scene']==scene) for scene in scene_names}
        for scope in scopes:
            ss={scene:merge([r['stats'][scope] for r in cr if r['scene']==scene]) for scene in scene_names}
            all_sums=merge(list(ss.values()));final=finalize(all_sums,total)
            final.update(bootstrap(ss,scene_totals),
                         scenes_with_pixels=sum(v['n']>0 for v in ss.values()),
                         flow_fields_with_pixels=sum(r['stats'][scope]['n']>0 for r in cr),sufficient_statistics=all_sums)
            summaries[c]['subsets'][scope]=final
            table.append(dict(condition=c,subset=scope,**{k:v for k,v in final.items() if k!='sufficient_statistics' and not k.endswith('_ci95')}))
            for scene,stats in ss.items():
                scene_table.append(dict(condition=c,subset=scope,scene=scene,**finalize(stats,scene_totals[scene]),**stats))
        for r in cr:
            for scope in scopes:
                pair_table.append(dict(condition=c,scene=r['scene'],camera=r['camera'],direction=r['direction'],source=r['source'],target=r['target'],
                                       subset=scope,**finalize(r['stats'][scope],H*W)))
    result=dict(done=True,model='public WAFT-DAv2-a2 Spring',config_sha256=CONFIG_SHA,
                scale=.5,tta=False,flow_storage_dtype='float16',calculation_dtype='float64',
                definition=cfg,conditions=summaries,model_inference=False,cpu_only=True)
    write_json(OUT/'SUMMARY.json',result)
    csv_write(OUT/'subsets.csv',table);csv_write(OUT/'pairs.csv',pair_table);csv_write(OUT/'scenes.csv',scene_table)
    write_json(OUT/'PREDICTION_MANIFEST.json',list(pred_files.values()))
    write_json(OUT/'IMAGE_MANIFEST.json',list(inputs.values()))
    lines=['# 公開 WAFT：官方污染下未改值像素的預測分歧', '',
           f'完成：{datetime.datetime.now().astimezone().isoformat(timespec="seconds")}。CPU-only；只讀既有dense預測與官方PNG。', '',
           '**本報告只屬於公開WAFT-DAv2-a2 Spring權重、0.5 scale、無TTA的提交版本。**',
           f'Config SHA256：`{CONFIG_SHA}`。權重SHA256：`{cfg["prediction_config"]["checkpoint_sha256"]}`。',
           '來源：`runs/submission_v1/predictions/`。各prediction讀取時均重新核對原始journal SHA256；兩眼journal與DONE標記也通過config雜湊檢查。未讀寫submissions/v1內的檔案，未修改runs/official_delta_v1、paper或其他實驗。', '',
           '## 1. 主表：兩幀同座標都未改值 vs 至少一幀被改值', '',
           '對每組光流來源幀t與目標幀t±1，U(x)要求兩張corrupted影像各自都與clean在同一原生座標x的三個RGB通道逐值相同；T=非U。沒有差值門檻、容差或推估污染mask。',
           'EPE分歧為`mean_U ||flow_corrupt − flow_clean||₂`；R為`sum_U ||flow_corrupt||₂ / sum_U ||flow_clean||₂`。單位為原生1920×1080影像px/frame。這是光流預測之間的分歧，不是對GT的EPE，也不是stereo disparity。', '',
           '| 污染 | U比例 % | U：EPE分歧 | U：幅度比 R | T：EPE分歧 | T：幅度比 R | U有像素的scene / flow fields |',
           '|---|---:|---:|---:|---:|---:|---|']
    for c,v in summaries.items():
        u=v['subsets']['both_untouched'];t=v['subsets']['both_touched']
        lines.append(f'| {c} | {u["fraction"]*100:.6f} | {number(u["epe"])} | {number(u["magnitude_ratio"])} | {number(t["epe"])} | {number(t["magnitude_ratio"])} | {u["scenes_with_pixels"]}/10；{u["flow_fields_with_pixels"]}/3960 |')
    lines+=['','每種污染包含全部10個test序列、兩眼、FW/BW共3960組光流；每種共8,211,456,000個像素觀測。主表按子集像素數池化，不把每張影像或每scene的比例／幅度比直接平均。空子集保留N=0、EPE與幅度比為N/A；clean幅度和為0時幅度比也是N/A。', '',
            '| 污染 | U像素觀測数 | U：clean平均幅度 | U：corrupt平均幅度 | T：clean平均幅度 | T：corrupt平均幅度 |','|---|---:|---:|---:|---:|---:|']
    for c,v in summaries.items():
        u=v['subsets']['both_untouched'];t=v['subsets']['both_touched']
        lines.append(f'| {c} | {u["pixel_observations"]:,} | {number(u["clean_mean_magnitude"])} | {number(u["corrupt_mean_magnitude"])} | {number(t["clean_mean_magnitude"])} | {number(t["corrupt_mean_magnitude"])} |')
    lines+=['','## 2. 定義敏感度：來源幀與模型輸入鄰域', '',
            '單張官方影像未改值比例：每種2000張不同PNG，各張只計一次；與主表的兩幀交集、FW/BW像素觀測加權不同。Source-only EPE／幅度比則按光流來源幀的未改值像素池化。', '',
            '| 污染 | 單張影像未改值比例 % | Source-only像素比例 % | Source-only EPE | Source-only R |',
            '|---|---:|---:|---:|---:|']
    for c,v in summaries.items():
        u=v['subsets']['source_untouched']
        lines.append(f'| {c} | {v["unique_image_rgb_identical_fraction"]*100:.6f} | {u["fraction"]*100:.6f} | {number(u["epe"])} | {number(u["magnitude_ratio"])} |')
    lines+=['','5×5控制：使用與提交相同的uint8 INTER_AREA縮為960×540，要求兩幀模型輸入的5×5鄰域逐值相同，且原生位置屬於U。這排除「原生單像素沒變，但縮放後局部输入已變」的直接混淆。此控制不重跑模型。', '',
            'Clean-correspondence是另一個以主表U為基礎的控制：要求來源位置的模型5×5鄰域未改值，並檢查clean預測的目標對應位置，其四個鄰接取樣像素的原生RGB與模型5×5鄰域都未改值；超出畫面者排除。它檢查目標對應位置而不是上列的目標同座標鄰域，因此與兩幀5×5控制不是巢狀子集。使用clean預測而非GT，也不能當成已知正確對應。', '',
            '| 污染 | 兩幀5×5控制比例 % | EPE | R | 對應位置控制比例 % | EPE | R |',
            '|---|---:|---:|---:|---:|---:|---:|']
    for c,v in summaries.items():
        u=v['subsets']['local5_untouched'];t=v['subsets']['correspondence_local5_untouched']
        lines.append(f'| {c} | {u["fraction"]*100:.6f} | {number(u["epe"])} | {number(u["magnitude_ratio"])} | {t["fraction"]*100:.6f} | {number(t["epe"])} | {number(t["magnitude_ratio"])} |')
    lines+=['','## 3. 全圖參考與不確定性', '',
            '| 污染 | 全圖EPE分歧 | 全圖clean幅度 | 全圖corrupt幅度 | 全圖R |','|---|---:|---:|---:|---:|']
    for c,v in summaries.items():
        u=v['subsets']['all']
        lines.append(f'| {c} | {number(u["epe"])} | {number(u["clean_mean_magnitude"])} | {number(u["corrupt_mean_magnitude"])} | {number(u["magnitude_ratio"])} |')
    lines+=['','全圖數字是dense像素平均；官方伺服器使用抽樣點，因此不要求最後幾位與官方打包HDF5完全一致。本輪未重新計算或修改submission檔。', '',
            '| 污染 | U EPE CI95 | U R CI95 | 有效bootstrap次數：EPE / R |','|---|---|---|---|']
    for c,v in summaries.items():
        u=v['subsets']['both_untouched']
        lines.append(f'| {c} | {interval(u["epe_ci95"])} | {interval(u["magnitude_ratio_ci95"])} | {u["epe_bootstrap_valid_draws"]} / {u["magnitude_ratio_bootstrap_valid_draws"]} |')
    lines+=['','CI以10個scene整塊重抽10000次（seed=20260910），保留場景內幀與兩眼相關性；每次用像素和重新算EPE與幅度比。有空子集／零幅度和的重抽不填0，只列有效重抽的條件式區間及有效次數。若有效像素只來自一個scene，EPE／R的跨scene CI不估計，避免把退化的零寬區間當成確定性。這描述固定test場景組成的變動，非独立污染紋理母體CI。', '',
            '## 4. 機制連結與論文使用範圍', '',
            '1. **Spatter：公開權重上的非局部運動衰減獲得直接支持。** U占81.717954%，EPE分歧6.9115、幅度比0.4458；改值區為7.0481、0.4349，兩區都嚴重衰減。模型5×5鄰域也不變時仍有27.651585%的像素、EPE6.8051、R0.4550。對應位置控制仍有15.865276%、EPE3.1246、R0.4945；這些控制均涵蓋10個scene。',
            '2. **Frost：不能沿用舊合成資料的大範圍未污染區崩塌結論。** U僅0.076900%、2個scene，EPE0.6449、R0.9430；全圖R0.4244不代表此小子集也同程度崩塌。鄰域／對應位置控制各只剩1個有效scene。',
            '3. **Rain：主表U有放大，但嚴格局部控制不再顯示相同幅度。** U的R1.5692；5×5與對應位置控制降至1.0021、1.0040，其R的scene CI均跨1。因此主表本身不足以宣稱rain也有同樣穩健的非局部機制。',
            '4. **Snow：U的放大仍可見，但局部控制代表區域很小。** U的R1.2687；5×5控制R1.6139、占0.024315%，對應位置控制R1.4577、占0.021494%，兩者各有9個有效scene。不能將這些小區域的比率當成全圖效果。',
            '5. **Zoom blur：U的R1.1042，但最嚴格控制樣本不足。** 5×5只剩15,214個像素觀測、2個scene；對應位置控制僅2,275個、1個scene，不能據此提出穩健的跨scene結論。', '',
            '既有純CPU影像分析：[BENCHMARK_CORRUPTION_ALIGNMENT_ZH.md](../../BENCHMARK_CORRUPTION_ALIGNMENT_ZH.md)。它在官方全部spatter／frost影像上確認`C=min(I+T,255)`的全域固定模板等價性；同一模板跨時間、左右眼與序列共用，全部像素重建0差異。該幾何結果不依賴WAFT或V4 checkpoint，因此可以支撐本次公開權重分析的靜態貼圖背景。',
            'Frost幾乎滿版改值；剩餘bit-identical位置可能是clean已飽和成白色等特殊位置。不能將少量這類像素當成廣泛、具代表性的未污染背景，更不能沿用本地合成frost的61.61%比例。原生同座標兩幀未改值也不保證目標匹配位置／整個感受野未改變；因此另列縮放與clean-correspondence控制。',
            'U上出現預測分歧支持輸出變化不限於被改值的像素；本報告沒有test GT，因此不把分歧寫成真實錯誤增加倍數。實際結論應分污染、有效像素與控制結果陳述，不能自動把所有條件都稱為非局部崩塌。', '',
            '重現：本目錄analyze.py與summarize.py；CONFIG.json固定來源，PREDICTION_MANIFEST.json記錄23760個實際讀取並重新驗證雜湊的dense檔，IMAGE_MANIFEST.json記錄12000張PNG。完整子集數據為SUMMARY.json、subsets.csv、scenes.csv、pairs.csv。沒有模型推論、訓練、GPU呼叫或投稿。']
    (OUT/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
    verification=dict(done=True,config_sha256=CONFIG_SHA,flow_comparisons=19800,prediction_files_hash_verified=23760,
                      png_inputs=12000,subset_partition_verified=True,cpu_only=True,model_inference=False,
                      output_directory=str(OUT),finished_at=datetime.datetime.now().astimezone().isoformat(timespec='seconds'))
    write_json(OUT/'FINAL_VERIFIED.json',verification)
    print(json.dumps({c:{'unique_image_untouched':v['unique_image_rgb_identical_fraction'],
                         'untouched':v['subsets']['both_untouched'],'touched':v['subsets']['both_touched'],
                         'local5':v['subsets']['local5_untouched'],'correspondence_local5':v['subsets']['correspondence_local5_untouched']}
                      for c,v in summaries.items()},indent=2))


if __name__=='__main__':main()
