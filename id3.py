# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import math


class ID3(object):
    class Node(object):
        def __init__(self, _id):
            self.id = _id
            self.attribute = None
            self.branches = {}

        def __str__(self):
            return self.attribute

        __repr__ = __str__

        @property
        def node_name(self):
            # 可视化时，每个node，必须要有独一无二的name
            return ''.join([self.attribute, str(self.id)])

        def add_branch_node(self, value, node):
            self.branches[value] = node

        def add_to_graph(self, graph):
            graph.node(self.node_name, self.__str__())
            for edge_name, branch_node in self.branches.items():
                branch_node.add_to_graph(graph)
                graph.edge(self.node_name, branch_node.node_name, label=str(edge_name))

    def __init__(self, data, target):
        """ 初始化 ID3

        :param data: DataFrame，训练数据集
        :param target: 训练数据集中的目标attribute
        """
        self.node_counter = 0 # 方便可视化
        self.root_node = None
        self.data = data
        self.target = target
        self.attribute_values = {}
        for attribute in data.columns:
            self.attribute_values[attribute] = data[attribute].unique()

    def _entropy(self, data, attribute):
        """计算某个attribute的熵，私有
        """
        value_freq = data[attribute].value_counts()
        data_entropy = 0.0
        N = len(data)
        for value, freq in value_freq.items():
            p = freq / N
            data_entropy += p * self._info(data, attribute, value)
        return data_entropy

    def _info(self, data, attribute, attribute_value):
        data = data[data[attribute] == attribute_value]
        target_value_freq = data[self.target].value_counts()
        data_info = 0.0
        N = len(data)
        for freq in target_value_freq.values:
            p = freq / N
            data_info -= p * math.log(p, 2)
        return data_info

    def make_up_counts(self, attribute, counts):
        new_counts = dict.fromkeys(self.attribute_values[self.target], 0.0)
        new_counts.update(counts)
        return new_counts

    def _make_decision_tree(self, data, node):
        """建立决策树，私有
        """
        # 当数据都为同一类数据时，直接返回
        if len(data[self.target].value_counts()) == 1:
            node.attribute = data[self.target].value_counts().index[0]
            return
        # 如果除了target外，已经没有其他attribute了，那也返回
        if len(data.columns) == 1:
            node.attribute = data[self.target].value_counts().argmax()
            return

        # 寻找熵最小的属性
        min_entropy = math.inf
        for attribute in data.columns:
            if attribute == self.target:
                continue
            temp_entropy = self._entropy(data, attribute)
            if temp_entropy < min_entropy:
                min_entropy = temp_entropy
                node.attribute = attribute

        for value in data[node.attribute].value_counts().index:
            branch_data = data[data[node.attribute] == value]
            branch_data = branch_data.drop(node.attribute, axis=1)
            branch_node = self._new_node()
            node.add_branch_node(value, branch_node)
            self._make_decision_tree(branch_data, branch_node)

    def run(self):
        self.root_node = self._new_node()
        self._make_decision_tree(self.data, self.root_node)

    def _new_node(self):
        self.node_counter += 1
        return self.Node(_id=self.node_counter)

    def render_decision_tree(self, filename):
        """渲染决策树，需要graphviz支持
        """
        if not self.root_node:
            raise ValueError('Tree not decided!')

        from graphviz import Digraph
        dot_graph = Digraph(comment="Decision Tree")
        self.root_node.add_to_graph(dot_graph)
        dot_graph.render(filename)

    def _predict(self, row, node, force):
        """实际进行递归预测的函数，私有
        """
        if node.branches:
            try:
                return self._predict(row, node.branches[row[node.attribute]], force)
            except KeyError as e:
                if force:
                    return self._predict(row, next(iter(node.branches.values())), force)
                raise e
        else:
            return node.attribute

    def predict(self, data, force=False):
        """预测测试数据集的target值

        :param data: DataFrame 测试数据集
        :param force: 默认为 False，如果为 True，则当branch都不符合测试数据的值时，任意挑选一个branch
        :return: 预测值，list
        """
        if not hasattr(data, 'iterrows'):
            data = pd.DataFrame([data])
        results = []
        for index, row in data.iterrows():
            results.append(self._predict(row, self.root_node, force=force))
        return results

    @staticmethod
    def score(predict_results,  actual_results):
        return sum(predict_results == actual_results) / len(predict_results)


if __name__ == '__main__':
    # 使用示例
    # 输入数据应该为 Pandas 所定义的 DataFrame。这里直接用利用 pd.read_csv 读取 csv文件为 DataFrame
    train_data = pd.read_csv('golf.csv')
    # 初始化 ID3 算法时需要提供训练数据集以及目标属性
    id3_solver = ID3(train_data, target='play')
    # 进行训练
    id3_solver.run()
    # 输出训练的到的决策树
    id3_solver.render_decision_tree('./dtree')
    # 这里仅仅是为了展示功能，所以直接把训练数据集当作了测试集，实际使用时要分离
    test_data = train_data
    # 提取出 test_data 中的实际结果后删除
    # 实际上也可以不删除，这里是为了排除 ID3 利用已有结果作弊的可能
    result = test_data['play'].values
    test_data.drop('play', axis=1, inplace=True)
    # 输入 test_data，获得预测
    predict = id3_solver.predict(test_data)
    # 比较预测与实际结果获得正确率
    accuracy = id3_solver.score(predict, result)
    print('The accuracy of the prediction of test data is {}'.format(accuracy))

    #  更多的例子可以查看 nursery.py 与 mushroom.py



