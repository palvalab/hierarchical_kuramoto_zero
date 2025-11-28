import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

def plot_dfa_with_linfit(x, y, ax):
    lin_fit_res = sp.stats.linregress(np.log(x), np.log(y))

    ax.scatter(x, y, alpha=0.5)

    line_x = np.geomspace(np.min(x), np.max(x), 100)
    log_predicted = lin_fit_res.slope * np.log(line_x) + lin_fit_res.intercept
    predicted = np.exp(log_predicted)

    label = f'DFA={lin_fit_res.slope:.2f}\nR={lin_fit_res.rvalue**2:.2f}'
    ax.plot(line_x, predicted, label=label, lw=2)

    ax.set_xscale('log')
    ax.set_yscale('log')
