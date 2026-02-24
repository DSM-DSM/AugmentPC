import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from itertools import product

metric = 'precision'  # Optional Metrics: SHD,TPR,FPR,F1,precision,recall
fig_path = f'../figure/pval_ensemble/{metric}_vs_ensemble.png'
# 预先定义所有可能的 CIT 方法及其颜色映射
fixed_cit_values = ['spearman', 'kci', 'gcm', 'classifier', 'dgan', 'cauchy ensemble', 'stouffer ensemble',
                    'fisher ensemble']  # 按需要添加所有可能的方法
cit_palette = sns.color_palette('Paired', len(fixed_cit_values))
cit_color_map = dict(zip(fixed_cit_values, cit_palette))

data_path = '../aug_pc_result/p_ensemble.xlsx'
df = pd.read_excel(data_path, sheet_name='raw')
cit_mask = df['cit'] == 'pval_ensemble'
df.loc[cit_mask, 'cit'] = df.loc[cit_mask, 'p_combination'] + ' ensemble'
mechanism_mask = df['causal_mechanism'] == 'polynomial'
df.loc[mechanism_mask,'causal_mechanism'] = 'Polynomial'

# 其余设置保持不变
dpi = 300
grid_info_font_size = 16
fig_x_range = [0.08, 0.95]
fig_y_range = [0.1, 0.9]
legend_x_position = 0.93
legend_y_position = 0.5
font_family = 'sans-serif'
font_style = 'italic'
font_weight = 'normal'
font_color = 'black'

legend_font_entry = FontProperties()
legend_font_entry.set_family(font_family)
legend_font_entry.set_style(font_style)
legend_font_entry.set_weight(font_weight)

legend_font_title = FontProperties()
legend_font_title.set_family(font_family)
legend_font_title.set_style(font_style)
legend_font_title.set_weight(font_weight)

inner_wspace = 0.1
inner_hspace = 0.1

fig = plt.figure(figsize=(20, 15))
noise_type = sorted(df['noise'].unique())
causal_mechanism = sorted(df['causal_mechanism'].unique())
row_index = list(product(causal_mechanism, noise_type))
col_index = [(10, 3), (50, 0.4), (50, 2)]
row_mapping = {row: i for i, row in enumerate(row_index)}
col_mapping = {col: i for i, col in enumerate(col_index)}
outer_grid = GridSpec(len(row_index), len(col_index), wspace=0.1, hspace=0.2)

for i, row in enumerate(row_index):
    r_mask = (df['causal_mechanism'] == row[0]) & (df['noise'] == row[1])
    for j, col in enumerate(col_index):
        j_mask = (df['nodes'] == col[0]) & (df['expected_degree'] == col[1])
        cit_mask = r_mask & j_mask
        df_filter = df.loc[cit_mask]
        if j == 0:
            fig.text(
                x=fig_x_range[0],
                y=fig_y_range[0] + (fig_y_range[1] - fig_y_range[0]) * (i + 0.5) / len(row_index),
                s=f'Mechanism: {row[0]}\nNoise: {row[1]}',
                ha='center',
                va='center',
                fontsize=grid_info_font_size,
                fontfamily=font_family,
                fontstyle=font_style,
                fontweight=font_weight,
                color=font_color,
                rotation=90
            )
        if i == 0:
            fig.text(
                x=fig_x_range[0] + (fig_x_range[1] - fig_x_range[0]) * (j + 0.5) / len(col_index),
                y=fig_y_range[1],
                s=f'Nodes: {col[0]}\nExpected Degree: {col[1]}',
                ha='center',
                va='center',
                fontsize=grid_info_font_size,
                fontfamily=font_family,
                fontstyle=font_style,
                fontweight=font_weight,
                color=font_color
            )
        inner_grid = GridSpecFromSubplotSpec(1, 1, subplot_spec=outer_grid[len(row_index) - i - 1, j],
                                             wspace=inner_wspace, hspace=inner_hspace)
        ax0 = plt.Subplot(fig, inner_grid[0])
        sns.boxplot(x='cit', y=metric, data=df_filter, hue='cit', palette=cit_color_map, ax=ax0,
                    legend=False, order=fixed_cit_values,
                    showmeans=True,
                    meanprops={'marker': 'o', 'markerfacecolor': 'white', 'markeredgecolor': 'black'},
                    flierprops=dict(
                        marker='*',
                        markerfacecolor='red',
                        markersize=5,
                        markeredgecolor='red'
                    ))
        ax0.set_xticklabels('')
        ax0.set_xlabel('')
        ax0.spines['top'].set_visible(False)
        ax0.spines['right'].set_visible(False)
        ax0.set_ylabel('')
        fig.add_subplot(ax0)

legend_elements = [plt.Line2D([0], [0], marker='s', color='w',
                              markerfacecolor=cit_color_map[cit], markersize=10,
                              label=cit) for cit in fixed_cit_values]
legend = fig.legend(handles=legend_elements,
                    title='',
                    loc='upper center',
                    bbox_to_anchor=(0.5, 0.07),
                    ncol=len(fixed_cit_values),
                    fontsize=24,
                    frameon=False,
                    prop=legend_font_entry,
                    title_fontproperties=legend_font_title,
                    handlelength=2.0,
                    handleheight=2.0,
                    markerscale=3.0,
                    handletextpad=2
                    )
for handle in legend.legend_handles:
    handle.set_markeredgecolor('black')
    handle.set_markeredgewidth(0.5)  # 设置色块边框宽度

legend_bbox = legend.get_window_extent().transformed(fig.transFigure.inverted())

fig.text(legend_bbox.x0 - 0.02, (legend_bbox.y0 + legend_bbox.y1) / 2, 'CIT Methods:',
         fontsize=14,
         fontfamily=font_family,
         fontstyle=font_style,
         fontweight=font_weight,
         color=font_color,
         ha='right',
         va='center')
# fig.show()
fig.savefig(fig_path, dpi=dpi, bbox_inches='tight')
plt.close(fig)
