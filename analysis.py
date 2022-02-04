#!/usr/bin/env python3
import csv
from collections import OrderedDict
from statistics import mean, stdev, median
import matplotlib.pyplot as plt
from itertools import product, combinations
import numpy as np
import seaborn as sns

BOUND_SCALE_AXIS = True    # CHANGE THIS OPTION TO REPRODUCE FIGURES WITH BOUND ON SAME SCALE.


# IMPORTANT: USE TYPE 1 FONTS FOR SUBMISSIONS.
plt.rc('text', usetex=True)
plt.rc('font', family='serif')



FLOAT_KEYS = ('Best Full Bound',
              'Test Error',
              'Train Error',
              'Norm Squared U',
              'Norm Squared V',
              'Number Parameters',
              'Best Prior Bound',
              'Best Prior Sigma U',
              'Best Prior Sigma V',
              'Stochastic Bnd Loss',
              'Stochastic Test Loss',
              'Beta',
              'Step Size',
              'Layer Width',
              'Batch Size',
              'Momentum',
              'CE Stopping Value',
              'Prior Examples',
              'Init Key',)


def rename_key(dic, old_key, new_key):
    "Rename a dictionary key."
    dic[new_key] = dic.pop(old_key)


def filter_results(list_of_dicts, search):
    """Filter list to those with bits matching search.

    e.g. filter_results([...], {"foo" = "bar",})
        => [{"foo" = "bar", "x" = 1}, {"foo" = "bar", "x" = 2}]
    """
    # Check for containment with dict1.items() >= dict2.items().
    return list(filter(lambda d: d.items() >= search.items(), list_of_dicts))


def group_by_hparam(list_of_dicts, x_label, xs):
    "Returns a list (sorted by xs) of list of dicts with same xs."
    return [filter_results(list_of_dicts, {x_label: x}) for x in xs]


def means_list_list_dict(ll_dicts, key):
    "Returns mean value of key from of sub-lists of dicts."
    return [mean(d[key] for d in ds) for ds in ll_dicts]


def stdev_list_list_dict(ll_dicts, key):
    "Returns mean value of key from of sub-lists of dicts."
    return [stdev(d[key] for d in ds) for ds in ll_dicts]


def pretty_dict_format(dic):
    return "  ".join([f"{k}: {v}" for k, v in dic.items()])



def get_results(filename):
    "Load the CSV file."
    with open(filename, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        results = [OrderedDict({k: v for k, v in row.items()}) for row in reader]

    for rec in results:
        for k in FLOAT_KEYS:
            rec[k] = float(rec[k])


    # Rename N Hidden to Width
    # for d in results:
    #     rename_key(d, "N Hidden", "Width")

    return results






def plot_double(ax, results, filters, xlabel, xs, gen_range=0.05, show_legend=True):
    "Plot Complexity and Generalisation vs xlabel for xs after applying filters."
    filtered = filter_results(results, filters)
    by_x = group_by_hparam(filtered, xlabel, xs)

    comp_means = means_list_list_dict(by_x, "Best Prior Bound")
    comp_stdevs = stdev_list_list_dict(by_x, "Best Prior Bound")
    gen_means = means_list_list_dict(by_x, "Test Error")
    gen_stdevs = stdev_list_list_dict(by_x, "Test Error")
    stoch_means = means_list_list_dict(by_x, "Stochastic Bnd Loss")
    stoch_stdevs = stdev_list_list_dict(by_x, "Stochastic Bnd Loss")

    error_kw = {"capsize": 5, "capthick": 1}
    fontsize_axes = 12
    ax.set_xscale("log")
    ax.set_xlabel(xlabel, fontsize=fontsize_axes)
    ax.set_title(pretty_dict_format(filters), fontsize=14)
    ax.tick_params(axis="x", which="minor")
    # ax.set_xticks([10**4, 6*10**4])
    # ax.set_xticklabels([10**4, 6*10**4])



    ax.set_ylabel("Coupled Bound", fontsize=fontsize_axes)
    comp = ax.errorbar(
        xs,
        comp_means,
        color="red",
        yerr=comp_stdevs,
        fmt=":o",
        label="Coupled Bound (L)",
        **error_kw)
    ax.set_ylim(0., 1.)



    right_ax = ax.twinx()
    right_ax.set_ylabel("Stochastic Error / Test Error", fontsize=fontsize_axes)
    if BOUND_SCALE_AXIS:
        right_ax.set_ylim(0., .5)
    else:
        right_ax.set_ylim(0., 1.)
    gen = right_ax.errorbar(
        xs,
        gen_means,
        color="green",
        yerr=gen_stdevs,
        fmt="--o",
        label="Test Error (R)",
        **error_kw)

    stochastic = right_ax.errorbar(
        xs,
        stoch_means,
        color="blue",
        yerr=stoch_stdevs,
        fmt=":o",
        label="Stochastic Error (R)",
        **error_kw)

    if show_legend:
        ax.legend([comp, gen, stochastic], [comp.get_label(), gen.get_label(), stochastic.get_label()])





def width_plot(results, widths, save_name=""):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.set_size_inches(12, 5, forward=True)
    fig.suptitle("Error and Bound Comparison by Width", fontsize=16)
    plot_double(ax1, results, {"Step Size": 0.1}, "Layer Width", widths)
    plot_double(ax2, results, {"Step Size": 0.01}, "Layer Width", widths)
    plot_double(ax3, results, {"Step Size": 0.001}, "Layer Width", widths)
    fig.tight_layout(pad=1.)    # Needs to be after plots.
    fig.savefig("figs/Figure_width-" + save_name + ".eps", format="eps")


def lr_plot(results, lrs, save_name=""):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.set_size_inches(12, 5, forward=True)
    fig.suptitle("Error and Bound Comparison by Learning Rate", fontsize=16)
    plot_double(ax1, results, {"Layer Width": 50}, "Step Size", lrs)
    plot_double(ax2, results, {"Layer Width": 200}, "Step Size", lrs)
    plot_double(ax3, results, {"Layer Width": 800}, "Step Size", lrs)
    fig.tight_layout(pad=1.)
    fig.savefig("figs/Figure_lr-" + save_name + ".eps", format="eps")







def best_prior_bound(results):
    best = min(results, key=lambda k: k['Best Prior Bound'])
    print(f"Test Error {best['Test Error']:.3f}  --  Full Bound {best['Best Full Bound']:.3f}  --  Prior Bound {best['Best Prior Bound']:.3f}")






def fourway_width_plot(widths, gelu_mnist, gelu_fashion, shel_mnist, shel_fashion, name):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    fig.set_size_inches(6.5, 6.5, forward=True)
    fig.suptitle("Error and Bound Comparison by Width", fontsize=16)
    plot_double(ax1, gelu_mnist, {"Step Size": 0.01}, "Layer Width", widths, show_legend=True)
    plot_double(ax2, gelu_fashion, {"Step Size": 0.01}, "Layer Width", widths, show_legend=False)
    plot_double(ax3, shel_mnist, {"Step Size": 0.01}, "Layer Width", widths, show_legend=False)
    plot_double(ax4, shel_fashion, {"Step Size": 0.01}, "Layer Width", widths, show_legend=False)
    ax1.set_title("GELU MNIST", fontsize=14)
    ax2.set_title("GELU Fashion", fontsize=14)
    ax3.set_title("SHEL MNIST", fontsize=14)
    ax4.set_title("SHEL Fashion", fontsize=14)
    fig.tight_layout(pad=1.)
    fig.savefig(name, format="eps", pad_inches=0.1)



def fourway_lr_plot(lrs, gelu_mnist, gelu_fashion, shel_mnist, shel_fashion, name):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    fig.set_size_inches(6.5, 6.5, forward=True)
    fig.suptitle("Error and Bound Comparison by Learning Rate", fontsize=16)
    plot_double(ax1, gelu_mnist, {"Layer Width": 200}, "Step Size", lrs, show_legend=True)
    plot_double(ax2, gelu_fashion, {"Layer Width": 200}, "Step Size", lrs, show_legend=False)
    plot_double(ax3, shel_mnist, {"Layer Width": 200}, "Step Size", lrs, show_legend=False)
    plot_double(ax4, shel_fashion, {"Layer Width": 200}, "Step Size", lrs, show_legend=False)
    ax1.set_title("GELU MNIST", fontsize=14)
    ax2.set_title("GELU Fashion", fontsize=14)
    ax3.set_title("SHEL MNIST", fontsize=14)
    ax4.set_title("SHEL Fashion", fontsize=14)
    fig.tight_layout(pad=1.)
    fig.savefig(name, format="eps", pad_inches=0.1)




def momentum_sgd():

    lrs = [0.1, 0.03, 0.01, 0.003, 0.001]
    widths = [50, 100, 200, 400, 800, 1600]
    tsizes = [60000,]

    results = get_results("results/grid-gelu-fashion.csv")
    gelu_fashion = filter_results(results, {"Model Type": "GELU", "Dataset": "fashion", "Momentum": 0.9})

    results = get_results("results/grid-shel-fashion.csv")
    shel_fashion = filter_results(results, {"Model Type": "SHEL", "Dataset": "fashion", "Momentum": 0.9})

    results = get_results("results/grid-gelu-mnist.csv")
    gelu_mnist = filter_results(results, {"Model Type": "GELU", "Dataset": "mnist", "Momentum": 0.9})

    results = get_results("results/grid-shel-mnist.csv")
    shel_mnist = filter_results(results, {"Model Type": "SHEL", "Dataset": "mnist", "Momentum": 0.9})


    print("\n\nGELU on Fashion-MNIST")

    print("Filtered Results:", len(gelu_fashion))
    best_prior_bound(gelu_fashion)

    width_plot(gelu_fashion, widths, "gelu-fashion")
    lr_plot(gelu_fashion, lrs, "gelu-fashion")



    print("\n\nSHEL on Fashion-MNIST")

    print("Filtered Results:", len(shel_fashion))
    best_prior_bound(shel_fashion)

    width_plot(shel_fashion, widths, "shel-fashion")
    lr_plot(shel_fashion, lrs, "shel-fashion")



    print("\n\nGELU on MNIST")

    print("Filtered Results:", len(gelu_mnist))
    best_prior_bound(gelu_mnist)

    width_plot(gelu_mnist, widths, "gelu-mnist")
    lr_plot(gelu_mnist, lrs, "gelu-mnist")



    print("\n\nSHEL on MNIST")

    print("Filtered Results:", len(shel_mnist))
    best_prior_bound(shel_mnist)

    width_plot(shel_mnist, widths, "shel-mnist")
    lr_plot(shel_mnist, lrs, "shel-mnist")

    fourway_width_plot(widths, gelu_mnist, gelu_fashion, shel_mnist, shel_fashion, "figs/Figure_width-all.eps")
    fourway_lr_plot(lrs, gelu_mnist, gelu_fashion, shel_mnist, shel_fashion, "figs/Figure_lrs-all.eps")


def vanilla_sgd():

    print("Vanilla SGD Table")
    results = get_results("results/grid-vanilla-sgd.csv")
    gelu_fashion = filter_results(results, {"Model Type": "GELU", "Dataset": "fashion", "Momentum": 0.0})
    shel_fashion = filter_results(results, {"Model Type": "SHEL", "Dataset": "fashion", "Momentum": 0.0})
    gelu_mnist = filter_results(results, {"Model Type": "GELU", "Dataset": "mnist", "Momentum": 0.0})
    shel_mnist = filter_results(results, {"Model Type": "SHEL", "Dataset": "mnist", "Momentum": 0.0})
    print("GELU Fashion")
    best_prior_bound(gelu_fashion)
    print("SHEL Fashion")
    best_prior_bound(shel_fashion)
    print("GELU MNIST")
    best_prior_bound(gelu_mnist)
    print("SHEL MNIST")
    best_prior_bound(shel_mnist)

    widths = [50, 100, 200, 400, 800, 1600]
    lrs = [0.1, 0.03, 0.01]
    fourway_width_plot(widths, gelu_mnist, gelu_fashion, shel_mnist, shel_fashion, "figs/Figure_width-all-vanilla.eps")
    fourway_lr_plot(lrs, gelu_mnist, gelu_fashion, shel_mnist, shel_fashion, "figs/Figure_lrs-all-vanilla.eps")



momentum_sgd()
vanilla_sgd()
plt.show()
