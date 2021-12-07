import numpy as np
import pandas as pd
import json


class ID3Tree():
    __count = 0

    def __init__(self):
        super().__init__()
        self.gain = {}

    # 计算数据集的熵
    def _entropy(self, dataSet):
        labels = list(dataSet.columns)
        level_count = dataSet[labels[-1]].value_counts().to_dict()  # 统计分类标签不同水平的值
        entropy = 0.0
        for key, value in level_count.items():
            prob = float(value) / dataSet.shape[0]
            entropy += -prob * np.log2(prob)
        return entropy

    # 获取子数据集
    def _split_dataSet(self, dataSet, column, level):
        subdata = dataSet[dataSet[column] == level]
        del subdata[column]  # 删除这个划分字段列
        return subdata.reset_index(drop=True)  # 重建索引

    # 计算每个分类标签的信息增益，返回最大信息增益对应的标签
    def _best_split(self, dataSet):
        best_info_gain = 0.0    # 求最大信息增益
        best_label = None       # 求最大信息增益对应的标签
        labels = list(dataSet.columns)[: -1]   # 不包括最后一个靶标签
        init_entropy = self._entropy(dataSet)  # 先求靶标签的香农熵
        for _, label in enumerate(labels):
            # 根据该label切割子数据集，并求熵
            levels = dataSet[label].unique().tolist()  # 获取该分类标签的不同level
            label_entropy = 0.0   # 用于累加各水平的信息熵；分类标签的信息熵等于该分类标签的各水平信息熵与其概率积的和。
            # 遍历所有分类标签
            for level in levels:
                level_data = dataSet[dataSet[label] == level]  # 获取该水平的数据集
                prob = level_data.shape[0] / dataSet.shape[0]  # 计算该水平的数据集在总数据集的占比
                # 计算熵，并更新到label_entropy中
                label_entropy += prob * self._entropy(level_data)  # _entropy用于计算香农熵

            # 计算信息增益
            info_gain = init_entropy - label_entropy

            # 用best_info_gain来取info_gain的最大值，并获取对应的分类标签
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_label = label

        self.__count += 1
        return best_label


    # 创建决策树
    def mktree(self, dataSet):
        target_list = dataSet.iloc[:, -1]
        # 子样本集只有单个属性
        if target_list.unique().shape[0] <= 1:
            return target_list[0]
        # 数据集只剩下把标签这一列数据；返回数量最多的水平
        if dataSet.shape[1] == 1:
            return self._top_amount_level(target_list)

        # 不满足终止条件时，递归处理
        best_label = self._best_split(dataSet)
        #递归结束

        best_label_levels = dataSet[best_label].unique().tolist()
        tree = {best_label: {}}  # 生成字典，用于保存树状分类信息；这里不能用self.tree = {}存储
        for level in best_label_levels:
            level_subdata = self._split_dataSet(dataSet, best_label, level)  # 获取该水平的子数据集
            tree[best_label][level] = self.mktree(level_subdata)  # 返回结果
        return tree

    def predict(self, tree, labels, test_sample):
        """
        对单个样本进行分类
        tree: 训练的字典
        labels: 除去最后一列的其它字段
        test_sample: 需要分类的一行记录数据
        """
        classLabel = None
        firstStr = list(tree.keys())[0]  # tree字典里找到第一个用于分类键值对
        secondDict = tree[firstStr]
        featIndex = labels.index(firstStr)  # 找到第一个建(label)在给定label的索引
        for key in secondDict.keys():
            if test_sample[featIndex] == key:  # 找到test_sample在当前label下的值
                if secondDict[key].__class__.__name__ == "dict":
                    classLabel = self.predict(secondDict[key], labels, test_sample)
                else:
                    classLabel = secondDict[key]
        return classLabel

    def _unit_test(self):
        data = [
            ['晴', '炎热', '高', '弱', '取消'],  # 1
            ['晴', '炎热', '高', '强', '取消'],  # 1
            ['阴', '炎热', '高', '弱', '进行'],  # 1
            ['雨', '适中', '高', '弱', '进行'],  # 1
            ['雨', '寒冷', '正常', '弱', '进行'],  # 1
            ['雨', '寒冷', '正常', '强', '取消'],  # 1
            ['阴', '寒冷', '正常', '强', '进行'],  # 1
            ['晴', '适中', '高', '弱', '取消'],  # 1
            ['晴', '寒冷', '正常', '弱', '进行'],  # 1
            ['雨', '适中', '正常', '弱', '进行'],  # 1
            ['晴', '适中', '正常', '强', '进行'],  # 1
            ['阴', '适中', '高', '强', '进行'],  # 1
            ['阴', '炎热', '正常', '弱', '进行'],  # 1
            ['雨', '适中', '高', '强', '取消'],  # 1
        ]
        data = pd.DataFrame(data=data, columns=['天气', '温度', '湿度', '风速', '活动'])
        # 生成树
        self.tree = self.mktree(data)
        print(self.tree)


        # 测试样本
        test_sample = ['晴', '炎热', '高', '若']
        # 预测结果
        outcome = self.predict(self.tree, ['天气', '温度', '湿度', '风速'], test_sample)
        print("The ID3Tree pridicet the game will " + outcome)


model = ID3Tree()
model._unit_test()
