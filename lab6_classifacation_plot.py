from pylab import *
import matplotlib.font_manager as mfm
import matplotlib.pyplot as plt

# 这里选取电脑中支持中文的字体，即可显示中文

font_path = "C:/Users/admin/AppData/Local/Microsoft/Windows/Fonts/FZFSK.TTF"
prop = mfm.FontProperties(fname=font_path)

# 结点和连接线的样式
decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")


def get_leaf_count(my_tree):
    """获取叶子结点的个数"""
    count = 0
    first_key = list(my_tree.keys())[0]
    second_dict = my_tree[first_key]

    for second_key in second_dict.keys():
        if isinstance(second_dict[second_key], dict):
            count += get_leaf_count(second_dict[second_key])
        else:
            count += 1
    return count


def get_tree_depth(my_tree):
    """获取树的高度"""
    max_depth = 0
    first_key = list(my_tree.keys())[0]
    second_dict = my_tree[first_key]

    for second_key in second_dict.keys():
        if isinstance(second_dict[second_key], dict):
            this_depth = 1 + get_tree_depth(second_dict[second_key])
        else:
            this_depth = 1

        if this_depth > max_depth:
            max_depth = this_depth

    return max_depth


def plot_node(node_txt, center_pt, parent_pt, node_type):
    """绘制树的节点"""
    plot.ax1.annotate(node_txt, xy=parent_pt, xycoords='axes fraction',
                      xytext=center_pt, textcoords='axes fraction',
                      va="center", ha="center", bbox=node_type, arrowprops=arrow_args, fontproperties=prop)


def plot_mid_text(cntr_pt, parent_pt, txt_string):
    """绘制线上的文字"""
    xMid = (parent_pt[0] - cntr_pt[0]) / 2.0 + cntr_pt[0]
    yMid = (parent_pt[1] - cntr_pt[1]) / 2.0 + cntr_pt[1]
    plot.ax1.text(xMid, yMid, txt_string, va="center", ha="center", rotation=30, fontproperties=prop)


def plot_tree(my_tree, parent_pt, node_txt):
    """绘制树形结构"""
    numLeafs = get_leaf_count(my_tree)
    firstStr = list(my_tree.keys())[0]
    cntrPt = (plot_tree.xOff + (1.0 + float(numLeafs)) / 2.0 / plot_tree.totalW, plot_tree.yOff)
    plot_mid_text(cntrPt, parent_pt, node_txt)
    plot_node(firstStr, cntrPt, parent_pt, decisionNode)
    secondDict = my_tree[firstStr]
    plot_tree.yOff = plot_tree.yOff - 1.0 / plot_tree.totalD
    for key in secondDict.keys():
        if isinstance(secondDict[key], dict):
            plot_tree(secondDict[key], cntrPt, str(key))
        else:
            plot_tree.xOff = plot_tree.xOff + 1.0 / plot_tree.totalW
            plot_node(secondDict[key], (plot_tree.xOff, plot_tree.yOff), cntrPt, leafNode)
            plot_mid_text((plot_tree.xOff, plot_tree.yOff), cntrPt, str(key))
    plot_tree.yOff = plot_tree.yOff + 1.0 / plot_tree.totalD


def plot(my_tree):
    """绘画"""
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    plot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plot_tree.totalW = float(get_leaf_count(my_tree))
    plot_tree.totalD = float(get_tree_depth(my_tree))
    plot_tree.xOff = -0.5 / plot_tree.totalW
    plot_tree.yOff = 1.0
    plot_tree(my_tree, (0.5, 1.0), '')
    plt.show()
