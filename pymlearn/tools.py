import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from .styler import Styler

def shuffle_split(x, y, test_size=0.33, scale_x=False, one_hot_y=False, seed=None):
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if one_hot_y:
        y = one_hot(y)
    elif y.ndim == 1:
        y = y.reshape(-1, 1)
    data = np.concatenate([x, y], axis=1)
    np.random.seed(seed)
    np.random.shuffle(data)
    split = round(test_size * data.shape[0])
    x_train = data[split:, :x.shape[1]]
    y_train = data[split:, x.shape[1]:]
    x_test = data[:split, :x.shape[1]]
    y_test = data[:split, x.shape[1]:]
    if scale_x:
        x_train, x_test = scale(x_train, x_test)
    return (x_train, y_train), (x_test, y_test)

def one_hot(y):
    y = np.array(y, dtype='int').flatten()
    n = y.size
    num_classes = np.max(y) + 1
    y_cat = np.zeros((n, num_classes))
    y_cat[np.arange(n), y] = 1
    return y_cat

def scale(x_train, x_test=None):
    x_train = np.array(x_train, dtype=float)
    x_test = np.array(x_test, dtype=float)
    if x_train.ndim == 1:
        x_train = x_train.reshape(-1, 1)
    u = np.nanmean(x_train, axis=0)
    s = np.nanstd(x_train, axis=0)
    x_train = (x_train - u) / s
    if not np.isnan(x_test).all():
        if x_test.ndim == 1:
            x_test = x_test.reshape(-1, 1)
        x_test = (x_test - u) / s
        return x_train, x_test
    else:
        return x_train

def plot_history(history, metric=None, save=False):
    styler = Styler()
    fig, ax = plt.subplots()
    if metric:
        for k, v in history.items():
            if metric[:3] in k:
                ax.plot(history[k], color=styler.color(k), linestyle=styler.linestyle(k))
        ax.set_ylabel(metric)
    else:
        ax2 = ax.twinx()
        for k, v in history.items():
            if 'loss' in k:
                ax.plot(history[k], color=styler.color(k), linestyle=styler.linestyle(k))
            else:
                ax2.plot(history[k], color=styler.color(k), linestyle=styler.linestyle(k))
        ax.set_ylabel('loss', color=styler.colors['primary'])
        ax2.set_ylabel('accuracy', color=styler.colors['secondary'])
    ax.set_title('model history')
    ax.set_xlabel('epoch')
    test_label = 'test'
    for k in history.keys():
        if 'val' in k:
            test_label = 'val'
            break
    train = Line2D([0], [0], color=styler.colors['legend'], linestyle=styler.linestyles['train'], label='train')
    test = Line2D([0], [0], color=styler.colors['legend'], linestyle=styler.linestyles['test'], label=test_label)
    fig.legend(handles=[train, test], loc='center')
    if save:
        fig.savefig('history.png')

def visualize_network(nodes, activations, save=False):
    styler = Styler()
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    num_params = 0
    for i in range(len(nodes)):
        x = np.ones(nodes[i]) + i
        bot_node = max(nodes) / 2 - nodes[i] / 2
        top_node = max(nodes) / 2 + nodes[i] / 2
        if i < len(nodes) - 1:
            bot_node_next = max(nodes) / 2 - nodes[i+1] / 2
            top_node_next = max(nodes) / 2 + nodes[i+1] / 2
            num_params += nodes[i] * nodes[i+1] + nodes[i+1]
        for j in np.arange(bot_node, top_node):
            ax.scatter(i+1, j, color=styler.colors['secondary'])
            if i < len(nodes) - 1:
                for k in np.arange(bot_node_next, top_node_next):
                    ax.plot([i+1, i+2], [j, k], color=styler.colors['primary'])
    ax.set_title('model architecture')
    ax.set_xlabel(f'{num_params} parameters')
    ax.set_ylabel('input')
    ax2.set_ylabel('output')
    ax.set_xticks(np.arange(2, len(nodes)+1))
    ax.set_xticklabels(activations)
    ax.set_yticks([])
    ax2.set_yticks([])
    if save:
        fig.savefig('architecture.png')
