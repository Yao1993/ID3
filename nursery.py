import pandas as pd
from id3 import ID3
import numpy as np

np.random.seed(1993)

# 读取所有数据
all_data = pd.read_csv('./nursery_data/all.csv')
# 利用permutation函数随机挑选1000个数据作为测试集，并将剩下的作为训练集
permutation = np.random.permutation(len(all_data))[:1000]
test_data = all_data.iloc[permutation]
result = test_data['classes'].values
test_data = test_data.drop('classes', axis=1)
train_data = all_data.drop(permutation)

id3_solver = ID3(train_data, target='classes')
id3_solver.run()
id3_solver.render_decision_tree('./nursery_data/dtree')

predict = id3_solver.predict(test_data, force=True)
accuracy = id3_solver.score(predict, result)
print('The accuracy of the prediction of test data is {}'.format(accuracy))
