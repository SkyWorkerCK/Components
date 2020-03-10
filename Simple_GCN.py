# coding:utf-8
import numpy as np
import tensorflow as tf
import RandomWalk
import matplotlib.pyplot as plt

# 邻接矩阵(有向图)
A = np.matrix([
    [0, 1, 0, 0],
    [0, 0, 1, 1],
    [0, 1, 0, 0],
    [1, 0, 1, 0]],
    dtype=float
)
# 基于每个节点的索引为其生成两个整数特征
"""
A*X = [[ 0.  0.]
     [ 1. -1.]
     [ 2. -2.]
     [ 3. -3.]]
     如果存在从 v 到 n 的边，则节点 n 是节点 v 的邻居
"""
X = np.matrix([
    [i, -i] for i in range(A.shape[0])
], dtype=float)

"""
    为了解决特征不包含自身的情况且梯度消失和梯度爆炸的情况，我们为其增加自环
"""
I = np.eye(A.shape[0])
A_hat = A + I

"""
    计算graph的度矩阵(计算入度————列)
    D**-1 * A = [[0.  1.  0.  0. ]
                 [0.  0.  0.5 0.5]
                 [0.  0.5 0.  0. ]
                 [1.  0.  1.  0. ]]
    采用传播规则：
    D**-1 * A * X = [[ 1.  -1. ]
                     [ 2.5 -2.5]
                     [ 0.5 -0.5]
                     [ 2.  -2. ]]
"""
D = np.array(np.sum(A, axis=0))[0]  # 变为一个list
D = np.matrix(np.diag(D))

"""
    ==================== 将以上整合 ==========================
    解决自环问题和梯度消失、梯度爆炸的情况
    np.linalg.inv(D_hat) * A_hat * X * W = [[ 1. -1.]
                                            [ 4. -4.]
                                            [ 2. -2.]
                                            [ 5. -5.]]
"""
W = np.matrix([
    [1, -1],
    [-1, 1]
])
A_hat = A + np.matrix(np.eye(A.shape[0]))
D_hat = np.diag(np.array(np.sum(A_hat, axis=0))[0])
end = tf.nn.relu(np.linalg.inv(D_hat) * A_hat * X * W)
print(end.numpy())

# ======================= Zachary 空手道俱乐部图网络 实例运用===========================================
import networkx as nx
from networkx import to_numpy_matrix


# 加载训练数据
def load_training_data(f_name):
    print("We are loading training data from " + f_name)
    edge_data_by_type = dict()
    all_nodes = list()
    with open(f_name, 'r') as f:
        for line in f:
            words = line[:-1].split(' ')
            # edge_data_by_type如果没有words[0]则针对它创建一个list
            if words[0] not in edge_data_by_type:
                edge_data_by_type[words[0]] = list()
            x, y = words[1], words[2]
            edge_data_by_type[words[0]].append((x, y))
            all_nodes.append(x)
            all_nodes.append(y)
    # 去掉重复元素
    all_nodes = list(set(all_nodes))
    print('Total training nodes: ' + str(len(all_nodes)))
    return edge_data_by_type, all_nodes


# 下面为模拟GCN部分
datasets = RandomWalk.pre_trained('./data')
zkc = nx.Graph()
_, all_nodes = load_training_data('./data/train.txt')

# 给所有节点排序
order = sorted(list(all_nodes))
# A为构建的邻接矩阵
A = to_numpy_matrix(zkc, nodelist=order)
# 给邻接矩阵强行加入自环
I = np.eye(A.shape[0])
A_hat = A + I
# 求A_hat的度矩阵（degree matrix）
D_hat = np.array(np.sum(A_hat, axis=0))[0]
D_hat = np.matrix(np.diag(D_hat))

# 随机初始化权重
W_1 = np.random.normal(loc=0, scale=1, size=(A.shape[0], 4))
W_2 = np.random.normal(loc=0, size=(W_1.shape[1], 2))


# 定义和堆叠GCN层
def gcn_layer(A_hat, D_hat, X, W):
    end = tf.nn.relu(np.linalg.inv(D_hat) * A_hat * X * W)
    return end.numpy()


# GCN 层
H_1 = gcn_layer(A_hat, D_hat, I, W_1)
H_2 = gcn_layer(A_hat, D_hat, H_1, W_2)
output = H_2

feature_representations = {
    node: np.array(output)[order.index(node)]
    for node in all_nodes
}
print(feature_representations)
# ======================= 基于得到的特征可视化===============================
print("values:\n")
print(feature_representations.values())

x = []
y = []
for key, values in feature_representations.items():
    x.append(values[0])
    y.append(values[1])

plt.figure()
plt.scatter(x, y, c='red')
plt.show()
