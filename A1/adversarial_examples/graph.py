#Graph Plotter

import os
import pandas as pd
import matplotlib.pyplot as plt
dfs = os.listdir("results")


def plot_graph(groups, method, attr, const_attr):
    fig, ax = plt.subplots()
    for name, group in groups:
        ax.plot(group[attr], group['Miss-classification Rate'], marker='o', linestyle='-', label=name)
    # set x-axis and y-axis labels and title
    ax.set_xlabel(attr)
    ax.set_ylabel('Miss-classification Rate')
    if const_attr:
      ax.set_title(f'{attr} vs. MSR at {const_attr}=0.05 and iters=40')
    else:
      ax.set_title(f'{attr} vs. MSR')
    ax.legend()
    #plt.show()
    plt.savefig(f"graphs/{method}_{attr.lower()}.png")


for i in range(len(dfs)):
    df_name = dfs[i].replace("_results_df.csv", "")
    df = pd.read_csv(f"results/{dfs[i]}", index_col=None)
    method = df["Method"][0]
    if method =="FGSM":
      groups = df.groupby('Model')
      plot_graph(groups, method, 'Epsilon', None)
    else:
      if "eps" in dfs[i]:
        eps_df = df.drop_duplicates().sort_values(by=['Epsilon'])
        eps_groups = eps_df.groupby('Model')
        plot_graph(eps_groups, method, 'Epsilon', "Alpha")
      else:
        alpha_df = df.drop_duplicates().sort_values(by=['Alpha'])
        alpha_groups = alpha_df.groupby('Model')
        plot_graph(alpha_groups, method, 'Alpha', 'Epsilon')
    print(f'{df_name} DONE')