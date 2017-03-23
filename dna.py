from id3 import ID3
import pandas as pd
import numpy as np

all_data = pd.read_csv('./dna_data/all.csv')
all_data = all_data.drop('name', axis=1)
all_data['dna'] = all_data['dna'].apply(lambda x: x.strip())
all_data['dna_len'] = all_data['dna'].apply(len)

columns = ['system']
for i in range(60):
    columns.append('d{}'.format(i))
modified_data = pd.DataFrame(columns=columns, index=all_data.index)

for index, row in all_data.iterrows():
    new_row = [row['system']]
    new_row.extend(list(row['dna']))
    modified_data.iloc[index] = new_row

permutation = np.random.permutation(len(modified_data))[:100]
test_data = modified_data.iloc[permutation]
result = test_data['system'].values
test_data = test_data.drop('system', axis=1)
train_data = modified_data.drop(permutation)

id3_solver = ID3(train_data, target='system')
id3_solver.run()
id3_solver.render_decision_tree('./dna_data/dtree')

predict = id3_solver.predict(test_data, force=True)
accuracy = id3_solver.score(predict, result)
print('The accuracy of the prediction of test data is {}'.format(accuracy))
