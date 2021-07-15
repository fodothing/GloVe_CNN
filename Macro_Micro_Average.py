# 计算多分类的宏平均
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np

def ma_mi_cro_avg(y_true, y_pred, eva_met):

    if len(eva_met) == 0:
        print("Not specified metric")

    # 标签值与预测值
    # y_true=[1, 2, 2, 3, 3]
    # y_pred=[1, 1, 3, 2, 3]

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    evaluation_metrics = eva_met  # "macro" or "micro"
    f1 = f1_score( y_true, y_pred, average=evaluation_metrics)
    p = precision_score(y_true, y_pred, average=evaluation_metrics)
    r = recall_score(y_true, y_pred, average=evaluation_metrics)

    # 取小数点5位
    f1 = round(f1, 5)
    p = round(p, 5)
    r = round(r, 5)

    print(f"{evaluation_metrics}-f1: {f1}")
