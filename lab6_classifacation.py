# coding=utf-8   #默认编码格式为utf-8

import csv
import operator
from math import log
from lab6_classifacation_plot import plot


def calc_entropy(data_set):
    """计算数据集的熵"""
    count = len(data_set)
    label_counts = {}

    # 统计数据集中每种分类的个数
    for row in data_set:
        label = row[-1]
        if label not in label_counts.keys():
            label_counts[label] = 1
        else:
            label_counts[label] += 1

    # 计算熵
    entropy = 0.0
    for key in label_counts:
        prob = float(label_counts[key]) / count
        entropy -= prob * log(prob, 2)
    return entropy


def calc_gini(data_set):
    """计算数据集的基尼值"""
    count = len(data_set)
    label_counts = {}

    # 统计数据集中每种分类的个数
    for row in data_set:
        label = row[-1]
        if label not in label_counts.keys():
            label_counts[label] = 1
        else:
            label_counts[label] += 1

    # 计算基尼值
    gini = 1.0
    for key in label_counts:
        prob = float(label_counts[key]) / count
        gini -= prob * prob
    return gini


def split_data_set(data_set, axis, value):
    """根据指定条件分割数据集"""
    # 划分后的新数据集
    new_data_set = []

    for row in data_set:
        if row[axis] == value:
            split_vector = row[:axis]
            split_vector.extend(row[axis + 1:])
            new_data_set.append(split_vector)
    return new_data_set


def choose_best_feature_1(data_set):
    """选取信息增益最高的特征"""
    feature_count = len(data_set[0]) - 1
    # 数据集的原始熵
    base_entropy = calc_entropy(data_set)
    # 最大的信息增益
    best_gain = 0.0
    # 信息增益最大的特征
    best_feature = -1

    # 遍历计算每个特征
    for i in range(feature_count):
        feature = [example[i] for example in data_set]
        feature_value_set = set(feature)
        new_entropy = 0.0

        # 计算信息增益
        for value in feature_value_set:
            sub_data_set = split_data_set(data_set, i, value)
            prob = len(sub_data_set) / float(len(data_set))
            new_entropy += prob * calc_entropy(sub_data_set)
        gain = base_entropy - new_entropy

        # 比较得出最大的信息增益
        if gain > best_gain:
            best_gain = gain
            best_feature = i

    return best_feature


def choose_best_feature_2(data_set):
    """根据增益率选取划分特征"""
    feature_count = len(data_set[0]) - 1
    # 数据集的原始熵
    base_entropy = calc_entropy(data_set)
    # 最大的信息增益率
    best_gain_ratio = 0.0
    # 信息增益率最大的特征
    best_feature = -1

    # 遍历计算每个特征
    for i in range(feature_count):
        feature = [example[i] for example in data_set]
        feature_value_set = set(feature)
        new_entropy = 0.0
        # 固有值
        intrinsic_value = 0.0

        # 计算信息增益
        for value in feature_value_set:
            sub_data_set = split_data_set(data_set, i, value)
            prob = len(sub_data_set) / float(len(data_set))
            new_entropy += prob * calc_entropy(sub_data_set)
            intrinsic_value -= prob * log(prob, 2)
        gain = base_entropy - new_entropy
        gain_ratio = gain / intrinsic_value

        # 比较得出最大的信息增益率
        if gain_ratio > best_gain_ratio:
            best_gain_ratio = gain_ratio
            best_feature = i

    return best_feature


def choose_best_feature_3(data_set):
    """基尼系数"""
    feature_count = len(data_set[0]) - 1
    # 最小基尼指数
    min_gini_index = 0.0
    # 基尼指数最小的特征
    best_feature = -1

    # 遍历计算每个特征
    for i in range(feature_count):
        feature = [example[i] for example in data_set]
        feature_value_set = set(feature)

        # 基尼指数
        gini_index = 0.0
        # 计算基尼指数
        for value in feature_value_set:
            sub_data_set = split_data_set(data_set, i, value)
            prob = len(sub_data_set) / float(len(data_set))
            gini_index += prob * calc_gini(sub_data_set)

        # 比较得出最小的基尼指数
        if gini_index < min_gini_index or min_gini_index == 0.0:
            min_gini_index = gini_index
            best_feature = i

    return best_feature


def get_top_class(labels):
    """从多个分类中选取出现频率最高的分类"""
    label_counts = {}
    for vote in labels:
        if vote not in label_counts.keys():
            label_counts[vote] = 0
        else:
            label_counts[vote] += 1
    sorted_label_count = sorted(label_counts.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sorted_label_count[0][0]


def create_division_tree(data_set, labels):
    """创建决策树"""
    class_list = [example[-1] for example in data_set]

    # 所有分类相同时返回
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]

    # 已经遍历完所有特征
    if len(data_set[0]) == 1:
        return get_top_class(class_list)

    # 选取最好的特征
    best_feat = choose_best_feature_3(data_set)
    best_feat_label = labels[best_feat]

    # 划分
    my_tree = {best_feat_label: {}}
    del (labels[best_feat])
    value_set = set([example[best_feat] for example in data_set])
    for value in value_set:
        sub_labels = labels[:]
        my_tree[best_feat_label][value] = create_division_tree(split_data_set(data_set, best_feat, value), sub_labels)
    return my_tree


def classify(division_tree, feat_labels, test_vector):
    """遍历决策树对测试数据进行分类"""
    first_key = list(division_tree.keys())[0]
    second_dict = division_tree[first_key]

    feat_index = feat_labels.index(first_key)
    test_key = test_vector[feat_index]

    test_value = second_dict[test_key]

    if isinstance(test_value, dict):
        class_label = classify(test_value, feat_labels, test_vector)
    else:
        class_label = test_value
    return class_label


def test():
    file_name = "adult+stretch.data"
    my_labels = ["颜色", "尺寸", "行为", "年龄"]

    with open(file_name, "r", encoding='utf-8') as file:
        my_data = list(csv.reader(file))
    my_tree = create_division_tree(my_data, my_labels)
    # print(my_data)
    plot(my_tree)


test()
