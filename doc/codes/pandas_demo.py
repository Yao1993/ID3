import pandas as pd

golf_data = pd.read_csv('../../golf.csv')

# 筛选出 outlook 为 sunny 的数据
golf_data[golf_data['outlook']=='sunny']
# 查看 outlook 各个值所具有的数据个数
golf_data['outlook'].value_counts()
# 丢弃 outlook 列
golf_data.drop('outlook', axis=1, inplace=True)
