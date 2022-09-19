# -*- coding:UTF-8 -*-
# modified from https://github.com/DaDaMrX/ReaLiSe
import argparse
import json
from collections import OrderedDict


def read_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        rows = [r.strip().split(', ') for r in f.read().splitlines()]

    data = []
    for row in rows:
        item = [row[0]]
        data.append(item)
        if len(row) == 2 and row[1] == '0':
            continue
        for i in range(1, len(row), 2):
            item.append((int(row[i]), row[i + 1]))
    # print(data)
    return data


def metric_file(pred_path, targ_path):
    preds = read_file(pred_path)
    targs = read_file(targ_path)
    metrics = OrderedDict()
    metrics["Detection"] = sent_metric_detect(preds=preds, targs=targs)
    metrics["Correction"] = sent_metric_correct(preds=preds, targs=targs)

    return metrics


def sent_metric_detect(preds, targs): # 句子级别 检测
    assert len(preds) == len(targs)
    tp, targ_p, pred_p, hit = 0, 0, 0, 0
    for pred_item, targ_item in zip(preds, targs):
        assert pred_item[0] == targ_item[0]
        pred, targ = sorted(pred_item[1:]), sorted(targ_item[1:])
        if targ != []: # 标注 + 1
            targ_p += 1
        if pred != []: # 识别 + 1
            pred_p += 1
        if len(pred) == len(targ) and all(p[0] == t[0] for p, t in zip(pred, targ)): # 完全正确
            hit += 1
        if pred != [] and len(pred) == len(targ) and all(p[0] == t[0] for p, t in zip(pred, targ)): # 预测的就正确
            tp += 1

    acc = hit / len(targs) # 完全正确
    p = tp / pred_p
    r = tp / targ_p
    f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0
    results = OrderedDict({
        'Accuracy': acc * 100,
        'Precision': p * 100,
        'Recall': r * 100,
        'F1': f1 * 100,
    })
    return results


def sent_metric_correct(preds, targs):
    assert len(preds) == len(targs)
    tp, targ_p, pred_p, hit = 0, 0, 0, 0
    for pred_item, targ_item in zip(preds, targs):
        assert pred_item[0] == targ_item[0]
        pred, targ = sorted(pred_item[1:]), sorted(targ_item[1:])
        if targ != []:
            targ_p += 1
        if pred != []:
            pred_p += 1
        if pred == targ:
            hit += 1
        if pred != [] and pred == targ:
            tp += 1

    acc = hit / len(targs)
    p = tp / pred_p
    r = tp / targ_p
    f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0
    results = OrderedDict({
        'Accuracy': acc * 100,
        'Precision': p * 100,
        'Recall': r * 100,
        'F1': f1 * 100,
    })
    return results


def get_sent_metrics(hyp, gold):
    metrics = metric_file(
        pred_path=hyp,
        targ_path=gold,
    )
    print("=" * 10 + " Sentence Level " + "=" * 10)
    for k, v in metrics.items():
        print(f"{k}: ")
        print(", ".join([f"{k_i}: {round(v_i, 2)}" for k_i, v_i in v.items()]))
    return metrics


if __name__ == "__main__":
    
    hyp_path = r"/home/hufei/pyapp/CCL2022/CCL2022-CLTC/C1_cuobiezi/datas/test_out/dev_model-train.v0.wang_and_sighan.lbl"
    # hyp_path = r"/home/hufei/pyapp/CCL2022/CCL2022-CLTC/C1_cuobiezi/cbz_models/f0_bert_base_v1/tmp/valid_27.lbl"
    hyp_path = r"/home/hufei/pyapp/CCL2022/CCL2022-CLTC/C1_cuobiezi/datas/test_out/temp.lbl"
    gold_path = r"/home/hufei/pyapp/CCL2022/CCL2022-CLTC/C1_cuobiezi/datas/track1/dev/yaclc-csc_dev.lbl"

    hyp_path = r"/home/hufei/pyapp/CCL2022/CCL2022-CLTC/C1_cuobiezi/src/metrics/samples/demo.hyp.labels"
    gold_path = r"/home/hufei/pyapp/CCL2022/CCL2022-CLTC/C1_cuobiezi/src/metrics/samples/demo.gold.labels"

    get_sent_metrics(hyp_path, gold_path)