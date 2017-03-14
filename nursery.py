import pandas as pd
from id3 import ID3
import numpy as np

np.random.seed(1993)

all_data = pd.read_csv('./nursery_data/all.csv')
permutation = np.random.permutation(len(all_data))[:1000]
test_data = all_data.iloc[permutation]
train_data = all_data.drop(permutation)

id3_solver = ID3(train_data, target='classes')
id3_solver.run()
id3_solver.render_decision_tree('./nursery_data/dtree')

predict = id3_solver.predict(test_data, force=True)
accuracy = id3_solver.score(predict, test_data['classes'].values)
print('The accuracy of test data is {}'.format(accuracy))
