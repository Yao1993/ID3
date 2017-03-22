# -*- coding: utf-8 -*-
from id3 import ID3
import pandas as pd

# 分别读取训练与测试数据
train_data = pd.read_csv('./mushroom_data/train.csv')
test_data = pd.read_csv('./mushroom_data/test.csv')
result = test_data['classes'].values
test_data = test_data.drop('classes', axis=1)

id3_solver = ID3(train_data, target='classes')
id3_solver.run()

id3_solver.render_decision_tree('./mushroom_data/dtree')

predict = id3_solver.predict(test_data)
accuracy = id3_solver.score(predict, result)
print('The accuracy of the prediction of test data is {}'.format(accuracy))
