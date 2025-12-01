__all__ = ['plot']

from matplotlib import pyplot as plt

def plot(_range,_value,kind,model_name,layer):
    """画图"""
    x = _range
    y = _value
    colors = ['r','g','b','k','y']
    labels = ['train','val','test']
    # kind = 'loss' | 'acc'

    # plt.subplot(1,2,1)
    for c,line in enumerate(y):
        plt.plot(x,line,color=colors[c],label=f'{kind} {labels[c]}',linewidth=0.7)

    plt.xlabel('epoch')
    plt.ylabel(f'{kind}')
    if kind == 'acc':
        plt.legend(loc=4)
    else:
        plt.legend(loc=1)
    plt.savefig(f'./{model_name}_{kind}.png',dpi=300)
    # plt.show()
    plt.close()
