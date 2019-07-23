import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, recall_score

"""
E.g. 

plt.rcParams['figure.figsize'] = [8, 8]
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14

"""


def calculate_recall_at_fpr(y_true, y_hat, fpr_target=0.04):
    fpr, tpr, thresholds = roc_curve(y_true, y_hat)
    fpr_idx = max(np.where(fpr <= fpr_target)[0])
    thresh = thresholds[fpr_idx]

    return recall_score(y_true, y_hat > thresh)


def plot_pr(y_trues, y_preds, labels):
    fig, ax = plt.subplots()
    for i, y_pred in enumerate(y_preds):
        y_true = y_trues[i]
        auc = roc_auc_score(y_true, y_pred)
        pr, re, thresholds = precision_recall_curve(y_true, y_pred)
        ax.plot(re, pr, label='%s; AUC=%.3f' % (labels[i], auc), marker='o', markersize=1)
    ax.legend()
    ax.grid()
    ax.set_title('Precision-Recall curve')
    ax.set_xlabel('Recall')
    _ = ax.set_ylabel('Precision')


def plot_roc(y_trues, y_preds, labels, x_max=1.0):
    fig, ax = plt.subplots()
    for i, y_pred in enumerate(y_preds):
        y_true = y_trues[i]
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        auc = roc_auc_score(y_true, y_pred)
        ax.plot(fpr, tpr, label='%s; AUC=%.3f' % (labels[i], auc), marker='o', markersize=1)

    ax.legend()
    ax.grid()
    ax.plot(np.linspace(0, 1, 20), np.linspace(0, 1, 20), linestyle='--')
    ax.set_title('ROC curve')
    ax.set_xlabel('False Positive Rate')
    ax.set_xlim([-0.01, x_max])
    _ = ax.set_ylabel('True Positive Rate')


def plot_hist(df, column, bins=20, normed=False, title=None):
    fig, ax = plt.subplots()
    _, bins_, _ = ax.hist(df[df.fraud_flag == 0][column], label='Legit', bins=bins, normed=normed, alpha=0.5)
    ax.hist(df[df.fraud_flag == 1][column], bins=bins_, label='Fraud', normed=normed, alpha=0.5)
    ax.legend()
    ax.grid()
    ax.set_xlabel('Model Score')
    if normed:
        ax.set_ylabel('Density')
    else:
        ax.set_ylabel('Count')
    if title is None:
        title = "Score distribution for %s" % column
    _ = ax.set_title(title)


def plot_losses(losses, labels):
    plt.figure()
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title('Evolution of training and testing losses')
    hs = list()
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

    for i, loss_set in enumerate(losses):
        train_1, test_1 = list(zip(*loss_set))
        label = labels[i]
        color = colors[i + 1 % len(colors)]
        x = range(len(train_1))
        h1, = plt.plot(x, train_1, '{c}'.format(c=color), marker='o', ms=3, label='%s train loss' % label)
        h2, = plt.plot(x, test_1, '{c}--'.format(c=color), marker='o', ms=3, label='%s dev loss' % label)
        hs += [h1, h2]

    l = plt.legend(handles=hs)
    _ = plt.grid()


def get_lr_opt(df_lr_opt, lr_opt_min_filter, lr_opt_max_filter):
    cols = ['train']
    if 'valid' in df_lr_opt:
        cols.append('valid')

    min_diff = df_lr_opt[
        (df_lr_opt.lr >= lr_opt_min_filter) &
        (df_lr_opt.lr <= lr_opt_max_filter)
        ].diff()[cols].mean(axis=1).min()

    lr_opt_min = df_lr_opt[df_lr_opt.diff()[cols].mean(axis=1) == min_diff].lr.values[0]

    min_loss = df_lr_opt[
        (df_lr_opt.lr >= lr_opt_min_filter) &
        (df_lr_opt.lr <= lr_opt_max_filter)
        ][cols].mean(axis=1).min()

    lr_opt_max = df_lr_opt[
        df_lr_opt[cols].mean(axis=1) == min_loss
        ].lr.values[0]

    lr_opt = 10 ** (np.mean([np.log10(lr_opt_max), np.log10(lr_opt_min)]))

    return lr_opt


def plot_lr_finder(df_lr_opt, lr_opt_min=None, lr_opt_max=None):
    if lr_opt_min:
        lr_opt = get_lr_opt(df_lr_opt, lr_opt_min, lr_opt_max)
    else:
        lr_opt = None

    cols = ['train']

    fig, ax = plt.subplots()

    if 'valid' in df_lr_opt:
        cols.append('valid')
        ax.scatter(x=df_lr_opt.lr, y=df_lr_opt.valid.values)
        ax.plot(df_lr_opt.lr, df_lr_opt.valid, label='Valid')

    ax.scatter(x=df_lr_opt.lr, y=df_lr_opt.train.values)
    ax.plot(df_lr_opt.lr, df_lr_opt.train, label='Train')

    y_max = df_lr_opt[cols].max().max() * 1.05
    y_min = df_lr_opt[cols].min().min() * .95

    if lr_opt:
        ax.vlines(lr_opt, y_min, y_max, linestyle='--', color='red', label='Optimum')

    ax.set_ylim([y_min, y_max])
    ax.semilogx()
    ax.grid()
    ax.legend()
    _ = ax.set_title('Loss vs Learning Rate')

    return lr_opt
