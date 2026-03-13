import numpy as np
import pandas as pd

data_path = 'p_ensemble&fdr_control.xlsx'
df = pd.read_excel(data_path, sheet_name='summary')
metric_l = ['F1', 'TPR', 'FPR', 'SHD', 'precision', 'recall']
param_l = [(10, 3), (50, 0.4), (50, 2)]

df_new_data = []
for p, expected_degree in param_l:
    mask = (df['nodes'] == p) & (df['expected_degree'] == expected_degree)
    df_slice = df.loc[mask, :]
    df_slice_group1 = df_slice.groupby(['causal_mechanism', 'noise'])
    for gid, data in df_slice_group1:
        for metric in metric_l:
            metric_mean = data[metric].str.extract(r'(\d+\.?\d*)', expand=False)
            metric_mean = pd.to_numeric(metric_mean, errors='coerce')
            data[f'{metric}_rank'] = metric_mean.rank(method="dense", ascending=False)
        df_new_data.extend(data.values)

rank_metric_col = [f'{m}_rank' for m in metric_l]
new_columns = list(df.columns) + rank_metric_col
df_slice_new = pd.DataFrame(df_new_data, columns=new_columns)

average_rank = []
for p, expected_degree in param_l:
    mask = (df['nodes'] == p) & (df['expected_degree'] == expected_degree)
    df_slice = df_slice_new.loc[mask, :]
    df_slice_group = df_slice.groupby(['cit', 'p_combination', 'fdr_alpha'])
    for gid, data in df_slice_group:
        average_rank_data = [p, expected_degree] + list(gid) + list(data.loc[:, rank_metric_col].mean().round(2))
        average_rank.append(average_rank_data)

average_rank_columns = ['节点数', '期望边数', 'CIT', 'p值集成方式', '可容忍错误发现率'] + [f'{m}平均秩' for m in metric_l]
df_average_rank = pd.DataFrame(average_rank, columns=average_rank_columns)
df_average_rank.loc[df_average_rank['CIT'] == 'pval_ensemble', 'CIT'] = df_average_rank.loc[df_average_rank['CIT'] == 'pval_ensemble', 'p值集成方式'] + ' ensemble'
df_average_rank.loc[:,'期望边数'] = df_average_rank['期望边数'] * df_average_rank['节点数']
df_average_rank.to_excel('../aug_pc_result/p_ensemble&fdr_control_average_rank.xlsx', index=False)
