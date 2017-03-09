from id3 import ID3

import pandas as pd

train_data = pd.read_csv('./mushroom_data/train.csv')
id3_solver = ID3(train_data, target='classes')
id3_solver.run()

id3_solver.render_decision_tree('./mushroom_data/dtree')

test_data = pd.read_csv('./mushroom_data/test.csv')
predict = id3_solver.predict(test_data)
accuracy = id3_solver.score(predict, test_data['classes'])
print('The accuracy of test data is {}'.format(accuracy))