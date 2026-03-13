import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import matplotlib.pyplot as mlp

from visualization.p_value_ensemble import fig_path

mlp.rcParams['font.sans-serif'] = ['SimHei']
mlp.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.unicode_minus'] = False
# fisherz, spearman, kendall, robustQn, pcor, conditional_distance, kci, gcm, classifier, vt, d_separation, wgcm,
# lp, knn, gan, dgan, diffusion


save = True
node_l = [10, 50]
data = pd.read_excel('../result/benchmark.xlsx', sheet_name='raw')
data['noise'] = data['noise'].map({
    'Cauchy': 'cauchy',
    'GaussianMixture5': 'gm5'
})

for node in node_l:
    df = data.loc[data['nodes'] == node, :]
    df.loc[:, 'cit'] = df['cit'].replace('conditional_distance', 'cdcit')
    df = df.rename(columns={'time_spent': '运行时间'})
    # 预先定义所有可能的 CIT 方法及其颜色映射
    if node == 10:
        fixed_cit_values = ['fisherz', 'spearman', 'kendall', 'robustQn', 'kci', 'gcm', 'wgcm', 'classifier', 'lp',
                            'knn', 'dgan', 'cdcit', 'diffusion']  # 按需要添加所有可能的方法
    else:
        fixed_cit_values = ['fisherz', 'spearman', 'kendall', 'robustQn', 'kci', 'gcm', 'wgcm', 'classifier', 'lp',
                            'knn', 'dgan']  # 按需要添加所有可能的方法
    cit_palette = sns.color_palette('Paired', len(fixed_cit_values))
    cit_color_map = dict(zip(fixed_cit_values, cit_palette))
    noise_type = sorted(df['noise'].unique())
    root_variable_generator = sorted(df['root'].unique())
    metrics = ['运行时间', 'SHD Anti', 'SHD', 'F1', 'FPR', 'TPR']

    # 其余设置保持不变
    dpi = 300
    grid_info_font_size = 20
    fig_x_range = [0.09, 0.95]
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

    causal_mechanisms_l = ['polynomial', 'nn']  # Polynomial, Neural Network
    for causal_mechanism in causal_mechanisms_l:
        for noise in noise_type:
            for root in root_variable_generator:
                fig = plt.figure(figsize=(4 * len(metrics), 18))
                outer_grid = GridSpec(len(metrics), 2, wspace=0.1, hspace=0.2)

                noise_root_mask = (df['noise'] == noise) & (df['root'] == root)
                df_filter = df[noise_root_mask].copy()

                for i, metric in enumerate(metrics):
                    fig.text(
                        x=fig_x_range[0],
                        y=fig_y_range[0] + (fig_y_range[1] - fig_y_range[0]) * (i + 0.5) / len(metrics),
                        s=metric,
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
                            x=fig_x_range[0] + (fig_x_range[1] - fig_x_range[0]) * .5,
                            y=fig_y_range[1],
                            s=f'因果机制：{causal_mechanism}，噪声分布：{noise}',
                            ha='center',
                            va='center',
                            fontsize=grid_info_font_size,
                            fontfamily=font_family,
                            fontstyle=font_style,
                            fontweight=font_weight,
                            color=font_color
                        )
                    inner_grid = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer_grid[len(metrics) - i - 1, :],
                                                         wspace=inner_wspace, hspace=inner_hspace)
                    expect_degrees = sorted(df_filter['expected_degree'].unique())

                    # 获取当前数据中存在的 CIT 方法，并按固定顺序排序
                    current_cits = [cit for cit in fixed_cit_values if cit in df_filter['cit'].unique()]

                    if metric == 'SHD' or metric == 'SHD Anti' or metric == '运行时间':
                        ax0 = plt.Subplot(fig, inner_grid[0])
                        mask0 = (df_filter['expected_degree'] == expect_degrees[0])
                        sns.boxplot(x='cit', y=metric, data=df_filter[mask0], hue='cit', palette=cit_color_map, ax=ax0,
                                    legend=False, order=current_cits,
                                    showmeans=True,
                                    meanprops={'marker': 'o', 'markerfacecolor': 'white', 'markeredgecolor': 'black'},
                                    flierprops=dict(
                                        marker='*',
                                        markerfacecolor='red',
                                        markersize=5,
                                        markeredgecolor='red'
                                    ))

                        # 为Time指标设置对数坐标轴
                        if metric == '运行时间':
                            ax0.set_yscale('log')
                            # 设置y轴刻度为10的幂次
                            ax0.yaxis.set_major_locator(plt.LogLocator(base=10, numticks=15))

                        ax0.set_xticklabels('')
                        ax0.set_xlabel('')
                        ax0.set_ylabel('')
                        text0 = '稀疏图' if expect_degrees[0] == min(expect_degrees) else '稠密图'
                        ax0.text(
                            x=0.5,
                            y=-0.05,
                            s=text0,
                            ha='center',
                            va='top',
                            transform=ax0.transAxes,
                            fontsize=14,
                            fontfamily=font_family,
                            fontstyle=font_style,
                            fontweight=font_weight,
                            color=font_color
                        )
                        ax0.spines['top'].set_visible(False)
                        ax0.spines['right'].set_visible(False)
                        fig.add_subplot(ax0)

                        ax1 = plt.Subplot(fig, inner_grid[1])
                        mask1 = (df_filter['expected_degree'] == expect_degrees[1])
                        sns.boxplot(x='cit', y=metric, data=df_filter[mask1], hue='cit', palette=cit_color_map, ax=ax1,
                                    legend=False, order=current_cits,
                                    showmeans=True,
                                    meanprops={'marker': 'o', 'markerfacecolor': 'white', 'markeredgecolor': 'black'},
                                    flierprops=dict(
                                        marker='*',
                                        markerfacecolor='red',
                                        markersize=5,
                                        markeredgecolor='red'
                                    ))

                        # 为Time指标设置对数坐标轴
                        if metric == '运行时间':
                            ax1.set_yscale('log')
                            # 设置y轴刻度为10的幂次
                            ax1.yaxis.set_major_locator(plt.LogLocator(base=10, numticks=15))

                        ax1.set_xticklabels('')
                        ax1.set_xlabel('')
                        ax1.set_ylabel('')
                        text1 = '稀疏图' if expect_degrees[1] == min(expect_degrees) else '稠密图'
                        ax1.text(
                            x=0.5,
                            y=-0.05,
                            s=text1,
                            ha='center',
                            va='top',
                            transform=ax1.transAxes,
                            fontsize=14,
                            fontfamily=font_family,
                            fontstyle=font_style,
                            fontweight=font_weight,
                            color=font_color
                        )
                        ax1.spines['top'].set_visible(False)
                        ax1.spines['right'].set_visible(False)
                        fig.add_subplot(ax1)
                    else:
                        ax0 = plt.Subplot(fig, inner_grid[0])
                        mask0 = (df_filter['expected_degree'] == expect_degrees[0])
                        sns.boxplot(x='cit', y=metric, data=df_filter[mask0], hue='cit', palette=cit_color_map, ax=ax0,
                                    legend=False, order=current_cits,
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
                        text0 = '稀疏图' if expect_degrees[0] == min(expect_degrees) else '稠密图'
                        ax0.text(
                            x=0.5,
                            y=-0.05,
                            s=text0,
                            ha='center',
                            va='top',
                            transform=ax0.transAxes,
                            fontsize=14,
                            fontfamily=font_family,
                            fontstyle=font_style,
                            fontweight=font_weight,
                            color=font_color
                        )
                        ax0.spines['top'].set_visible(False)
                        ax0.spines['right'].set_visible(False)
                        ax0.set_ylabel('')
                        fig.add_subplot(ax0)

                        ax1 = plt.Subplot(fig, inner_grid[1], sharey=ax0)
                        mask1 = (df_filter['expected_degree'] == expect_degrees[1])
                        sns.boxplot(x='cit', y=metric, data=df_filter[mask1], hue='cit', palette=cit_color_map, ax=ax1,
                                    legend=False, order=current_cits,
                                    showmeans=True,
                                    meanprops={'marker': 'o', 'markerfacecolor': 'white', 'markeredgecolor': 'black'},
                                    flierprops=dict(
                                        marker='*',
                                        markerfacecolor='red',
                                        markersize=5,
                                        markeredgecolor='red'
                                    ))
                        ax1.set_xticklabels('')
                        ax1.set_xlabel('')
                        ax1.set_ylabel('')
                        plt.setp(ax1.get_yticklabels(), visible=False)
                        text1 = '稀疏图' if expect_degrees[1] == min(expect_degrees) else '稠密图'
                        ax1.text(
                            x=0.5,
                            y=-0.05,
                            s=text1,
                            ha='center',
                            va='top',
                            transform=ax1.transAxes,
                            fontsize=14,
                            fontfamily=font_family,
                            fontstyle=font_style,
                            fontweight=font_weight,
                            color=font_color
                        )
                        ax1.spines['top'].set_visible(False)
                        ax1.spines['right'].set_visible(False)
                        ax1.spines['left'].set_visible(False)
                        ax1.tick_params(left=False, labelleft=False)
                        fig.add_subplot(ax1)

                # 创建图例元素，按照fixed_cit_values的顺序，但只包含当前数据中存在的CIT方法
                existing_cits = [cit for cit in fixed_cit_values if cit in df_filter['cit'].unique()]
                legend_elements = [plt.Line2D([0], [0], marker='s', color='w',
                                              markerfacecolor=cit_color_map[cit], markersize=10,
                                              label=cit) for cit in existing_cits]

                legend = fig.legend(handles=legend_elements,
                                    title='',
                                    loc='upper center',
                                    bbox_to_anchor=(0.5, 0.07),
                                    ncol=len(existing_cits),
                                    fontsize=24,
                                    frameon=False,
                                    prop=legend_font_entry,
                                    title_fontproperties=legend_font_title,
                                    handlelength=2.0,
                                    handleheight=2.0,
                                    markerscale=3.0,
                                    handletextpad=2
                                    )

                # 为每个图例句柄单独设置边框
                for handle in legend.legend_handles:
                    handle.set_markeredgecolor('black')
                    handle.set_markeredgewidth(0.5)  # 设置色块边框宽度

                legend_bbox = legend.get_window_extent().transformed(fig.transFigure.inverted())

                fig.text(legend_bbox.x0 - 0.02, (legend_bbox.y0 + legend_bbox.y1) / 2, 'CIT方法：',
                         fontsize=14,
                         fontfamily=font_family,
                         fontstyle=font_style,
                         fontweight=font_weight,
                         color=font_color,
                         ha='right',
                         va='center')
                fig_name = f'{causal_mechanism}_{node}_{noise}.png'
                print(fig_name)
                fig_path = f'../figure/benchmark/{fig_name}'
                if save:
                    fig.savefig(fig_path, dpi=dpi, bbox_inches='tight')
                    plt.close(fig)
                else:
                    plt.show()
