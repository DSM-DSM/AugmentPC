import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from itertools import product

data_path = '../aug_pc_result/fdr_control.xlsx'
df = pd.read_excel(data_path, sheet_name='raw')
mechanism_mask = df['causal_mechanism'] == 'polynomial'
df.loc[mechanism_mask, 'causal_mechanism'] = 'Polynomial'

metrics = ['F1', 'TPR', 'FPR', 'SHD']
fig_path = f'../figure/fdr_control/bh_bc_hyb.png'
# 预先定义所有可能的 CIT 方法及其颜色映射
procedure = ['BCProcedure', 'BHProcedure', 'HybAdaProcedure']
procedure_palette = sns.color_palette('Paired', len(procedure))
procedure_color_map = dict(zip(procedure, procedure_palette))

# 其余设置保持不变
dpi = 300
grid_info_font_size = 20
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

inner_wspace = 0.2
inner_hspace = 0.25

fig = plt.figure(figsize=(32, 24))
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
        mask = r_mask & j_mask
        df_filter = df.loc[mask]
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
        inner_grid = GridSpecFromSubplotSpec(2, 2, subplot_spec=outer_grid[len(row_index) - i - 1, j],
                                             wspace=inner_wspace, hspace=inner_hspace)
        pc_mask = df_filter['algo'] == 'pc'

        for k, metric in enumerate(metrics):
            df_filter2 = df_filter.loc[~pc_mask, :]
            ax = fig.add_subplot(inner_grid[k])
            sns.boxplot(x='fdr_alpha', y=metric, data=df_filter2, hue='procedure',
                        palette=procedure_palette, ax=ax, legend=False, showmeans=True,
                        meanprops={'marker': 'o', 'markerfacecolor': 'white', 'markeredgecolor': 'black'},
                        flierprops=dict(
                            marker='*',
                            markerfacecolor='red',
                            markersize=5,
                            markeredgecolor='red'
                        ))
            df_filter3 = df_filter.loc[pc_mask, :]
            mean_val = float(df_filter3[metric].mean())
            # 计算 5% 和 95% 分位数
            lower = df_filter3[metric].quantile(0.05)  # 5% 分位数
            upper = df_filter3[metric].quantile(0.95)  # 95% 分位数

            # 填充浅绿色区域（在绘制上下界之前或之后均可，通常先填充再画线，使线更清晰）
            x_min, x_max = ax.get_xlim()
            ax.fill_between([x_min, x_max], lower, upper, color='pink', alpha=0.3)

            # 绘制均值线和上下界线
            ax.axhline(y=mean_val, color='red', linestyle='--', linewidth=2, label='原始均值')
            ax.axhline(y=lower, color='gray', linestyle=':', linewidth=1.5, alpha=0.8)
            ax.axhline(y=upper, color='gray', linestyle=':', linewidth=1.5, alpha=0.8)

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_ylabel('')
            ax.set_xlabel(str(metric))
            fig.add_subplot(ax)

legend_elements = [plt.Line2D([0], [0], marker='s', color='w',
                              markerfacecolor=procedure_color_map[p], markersize=10,
                              label=p) for p in procedure]
legend = fig.legend(handles=legend_elements,
                    title='',
                    loc='upper center',
                    bbox_to_anchor=(0.5, 0.07),
                    ncol=len(procedure),
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

fig.text(legend_bbox.x0 - 0.02, (legend_bbox.y0 + legend_bbox.y1) / 2, 'FDR Control Methods:',
         fontsize=24,
         fontfamily=font_family,
         fontstyle=font_style,
         fontweight=font_weight,
         color=font_color,
         ha='right',
         va='center')
# fig.show()
fig.savefig(fig_path, dpi=dpi, bbox_inches='tight')
plt.close(fig)
