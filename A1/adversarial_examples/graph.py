#Graph Plotter

import pandas as pd
import matplotlib.pyplot as plt

def plot_graph(groups, method, const_attr):
    fig, ax = plt.subplots()
    for name, group in groups:
        ax.plot(group[const_attr], group['Miss-classification Rate'], marker='o', linestyle='-', label=name)
    # set x-axis and y-axis labels and title
    ax.set_xlabel(const_attr)
    ax.set_ylabel('Miss-classification Rate')
    ax.set_title(f'{const_attr} vs.Miss-classification Rate')
    ax.legend()
    #plt.show()
    plt.savefig(f"graphs/{method}_{const_attr.lower()}.png")

methods = ["fgsm", "pgd", "pgd-linf", "pgd-linf-targ", "pgd-l2"]
dfs = ['fgsm_results_df.csv', 'pgd_results_df.csv', 'pgd_linf_results_df.csv']

for i in range(len(dfs)):
    method = methods[i]
    df = pd.read_csv(f"results/{dfs[i]}", index_col=None)
    if method =="fgsm":
      groups = df.groupby('Model')
      plot_graph(groups, method, 'Epsilon')
    else:
      alpha_df = df[df['Alpha']==0.050].drop_duplicates().sort_values(by=['Epsilon'])
      alpha_groups = alpha_df.groupby('Model')
      plot_graph(alpha_groups, method, 'Epsilon')
      eps_df = df[df['Epsilon']==0.050].drop_duplicates().sort_values(by=['Alpha'])
      eps_groups = eps_df.groupby('Model')
      plot_graph(eps_groups, method, 'Alpha')