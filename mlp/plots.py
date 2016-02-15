import random
from itertools import chain

col_func = (lambda x: "#%06x" % random.randint(0, 0xFFFFFF))


def plot_stats(stats, mode='train_acc', shds=None, corr=False, max_epochs=30, figsize=(15, 25)):
    title = str(mode)
    mode_dict = {'train_acc': (0, 1),
                 'val_acc': (1, 1),
                 'test_cost': (2, 0),
                 'test_acc': (2, 1)}
    if not corr:
        mode = mode_dict[mode]
        plt.figure(figsize=(10, 10))
    else:
        fig, ax = plt.subplots(
            len(stats), figsize=figsize, sharex=True, sharey=True)
        ax = [ax]
        fig.suptitle(title)
    a = list()
    colors = ["#%06x" % random.randint(0, 0xFFFFFF)for i in range(len(stats))]

    for j, plo in enumerate(stats):

        if not corr:
            a.append(plt.plot(range(len(numpy.array(plo[mode[0]])[:, mode[1]][:max_epochs])),
                              numpy.array(plo[mode[0]])[:, mode[1]][:max_epochs], '-o', ms=2))
        else:
            mode = mode_dict[corr[0]]
            mode2 = mode_dict[corr[1]]

            train_cost = numpy.array(plo[mode[0]])[1:, mode[1]][1:max_epochs]
            test_cost = numpy.array(plo[mode2[0]])[1:, mode2[1]][1:max_epochs]

            # euclidean distance of each cost pair from origin: (a metric of
            # their combined error)
            euclid = np.sqrt(test_cost*test_cost + train_cost*train_cost)
            mx_ts = numpy.max(numpy.log(test_cost))
            mn_ts = numpy.min(numpy.log(test_cost))
            # Discretization gets carried out in the next two lines out of our
            # euclid error metric for the bubble plot
            hz, bi = numpy.histogram(euclid, bins=65)
            sz = list(chain(*[[20*c]*hz[c-1] for c in range(1, len(hz)+1)]))

            norm = numpy.sum(euclid)

            label = ax[j].scatter((train_cost),
                                  (test_cost),
                                  s=sz, alpha=0.5, color=colors[j])
            ax[j].set_xlabel("$log($"+title.split("vs")[-1]+"$)$")
            ax[j].set_ylabel("$log($"+title.split("vs")[0]+"$)$")

    if not corr:
        plt.legend(map(lambda x: x[0], a), shds, loc=5)
        plt.xlabel("epoch number")
        plt.ylabel(title.split("_")[-1])

    plt.show()
