# coding:utf-8
"""
    在已知“边多类型”网络上随机游走
    作者：陈珂
"""
import tensorflow as tf
import argparse
import random
from collections import defaultdict
import networkx as nx
import numpy as np
from gensim.models.keyedvectors import Vocab
from six import iteritems
import tqdm
import math
import time
from numpy import random

# 定义随机游走过程类
class RWGraph():
    def __init__(self, nx_G, node_type=None):
        self.G = nx_G
        self.node_type = node_type

    def walk(self, walk_length, start, schema=None):
        # Simulate a random walk starting from start node.
        G = self.G

        rand = random.Random()

        if schema:
            schema_items = schema.split('-')
            assert schema_items[0] == schema_items[-1]

        walk = [start]
        while len(walk) < walk_length:
            cur = walk[-1]
            candidates = []
            for node in G[cur].keys():
                if schema == None or self.node_type[node] == schema_items[len(walk) % (len(schema_items) - 1)]:
                    candidates.append(node)
            if candidates:
                walk.append(rand.choice(candidates))
            else:
                break
        return [str(node) for node in walk]

    def simulate_walks(self, num_walks, walk_length, schema=None):
        # 整个网络G
        G = self.G
        walks = []
        nodes = list(G.nodes())
        # print('Walk iteration:')
        # 当为同质网络时，游走模式为None
        if schema is not None:
            schema_list = schema.split(',')
        # num_walks为一共要游走的次数
        for walk_iter in range(num_walks):
            random.shuffle(nodes)
            for node in nodes:
                if schema is None:
                    walks.append(self.walk(walk_length=walk_length, start=node))
                else:
                    for schema_iter in schema_list:
                        if schema_iter.split('-')[0] == self.node_type[node]:
                            walks.append(self.walk(walk_length=walk_length, start=node, schema=schema_iter))

        return walks


# 定义对命令中参数的解析
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='data/amazon',
                        help='Input dataset path')
    parser.add_argument('--feature', type=str, default=None,
                        help='Input node features')
    parser.add_argument('--epoch', type=int, default=100,
                        help='Number of epoch, default is 100')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Number of batch_size. Default is 64')
    parser.add_argument('--eval-type', type=str, default=all,
                        help='The edge types for evaluation')
    parser.add_argument('--schema', type=str, default=None,
                        help='The metapath schema (e.g., U-I-U, I-U-I).')
    parser.add_argument('--dimensions', type=int, default=200,
                        help='Number of dimensions. Default is 200.')
    parser.add_argument('--edge-dim', type=int, default=10,
                        help='Number of edge embedding dimensions. Default is 10')
    parser.add_argument('--att-dim', type=int, default=20,
                        help='Number of attention dimensions. Default is 20.')
    parser.add_argument('--walk-length', type=int, default=10,
                        help='Length of walk per source. Default is 10.')
    parser.add_argument('--num-walks', type=int, default=20,
                        help='Number of walks per source. Default is 20.')
    parser.add_argument('--window-size', type=int, default=5,
                        help='Context size for optimization. Default is 5.')
    parser.add_argument('--negative-samples', type=int, default=5,
                        help='Negative samples for optimization. Default is 5.')
    parser.add_argument('--neighbor-samples', type=int, default=10,
                        help='Neighbor samples for aggregation. Default is 10.')
    parser.add_argument('--patience', type=int, default=5,
                        help='Early stopping patience. Default is 5.')
    return parser.parse_args()

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
    return edge_data_by_type

# 加载节点类型
def load_node_type(f_name):
    print('We are loading node type from:', f_name)
    node_type = {}
    with open(f_name, 'r') as f:
        for line in f:
            items = line.strip().split()
            node_type[items[0]] = items[1]
    return node_type

# 采用networkx搭建一个网络（边都为同一类型），并返回此搭建好了的网络G
def get_G_from_edges(edges):
    edge_dict = dict()
    for edge in edges:
        edge_key = str(edge[0]) + '_' + str(edge[1])
        if edge_key not in edge_dict:
            edge_dict[edge_key] = 1
        else:
            edge_dict[edge_key] += 1
    tmp_G = nx.Graph()      # nx为networkx
    for edge_key in edge_dict:
        weight = edge_dict[edge_key]
        x = edge_key.split('_')[0]
        y = edge_key.split('_')[1]
        tmp_G.add_edge(x, y)
        tmp_G[x][y]['weight'] = weight
    return tmp_G

# 产生游走序列（walks）
def generate_walks(network_data, num_walks, walk_length, schema, file_name):
    if schema is not None:
        node_type = load_node_type(file_name + '/node_type.txt')
    else:
        node_type = None

    all_walks = []
    # 根据不同的键值对进行分类
    for layer_id in network_data:
        tmp_data = network_data[layer_id]
        # start to do the random walk on a layer

        layer_walker = RWGraph(get_G_from_edges(tmp_data))
        # layer_walks为多次游走后的多条游走序列
        layer_walks = layer_walker.simulate_walks(num_walks, walk_length, schema=schema)

        all_walks.append(layer_walks)

    print('Finish generating the walks')

    return all_walks

# 把all_walks中的所有节点作为词汇总量建立词表vocab，并返回词表vocab
def generate_vocab(all_walks):
    index2word = []
    raw_vocab = defaultdict(int)

    for walks in all_walks:
        for walk in walks:
            for word in walk:
                raw_vocab[word] += 1

    vocab = {}
    for word, v in iteritems(raw_vocab):
        vocab[word] = Vocab(count=v, index=len(index2word))
        index2word.append(word)


    # 按照每个单词出现的频次进行从大到小排序在vocab中
    index2word.sort(key=lambda word: vocab[word].count, reverse=True)
    for i, word in enumerate(index2word):
        vocab[word].index = i

    # vocab是一个按照walks中所有节点出现的频率从大到小排序后的单词表
    # index2word是节点集合
    return vocab, index2word


# 生成节点对(224, 330, 1)
# 224是节点i，330是节点i在vocab中的左右窗口中的节点引索
def generate_pairs(all_walks, vocab, window_size):
    pairs = []
    skip_window = window_size // 2  # 整除
    for layer_id, walks in enumerate(all_walks):
        for walk in walks:
            for i in range(len(walk)):
                for j in range(1, skip_window + 1):
                    # 节点i的左窗口节点集合是引索，是节点i在vocab中的左边的元素
                    if i - j >= 0:
                        pairs.append((vocab[walk[i]].index, vocab[walk[i - j]].index, layer_id))
                    # 节点i的右窗口节点集合是引索，是节点i在vocab中的右边的元素
                    if i + j < len(walk):
                        pairs.append((vocab[walk[i]].index, vocab[walk[i + j]].index, layer_id))
    return pairs


# 训练模型
def train_model(network_data, feature_dic, log_name):
    # 所有的在不同edge type网络上的游走序列all_walks(同质网络游走)
    all_walks = generate_walks(network_data, args.num_walks, args.walk_length, args.schema, file_name)
    print("在网络上的随机游走序列为：", all_walks)
    # vocab是一个按照walks中所有节点出现的频率从大到小排序后的单词表
    # index2word是节点集合
    vocab, index2word = generate_vocab(all_walks)

    # 获取训练节点对
    train_pairs = generate_pairs(all_walks, vocab, args.window_size)


# 主函数
if __name__ == '__main__':
    args = parse_args()
    # 确定数据集所在文件夹位置
    print(args)
    file_name = args.input
    # 把特征数据置为空
    feature_dic = None
    # 定义日志文件名
    log_name = file_name.split('/')[-1]
    training_data = load_training_data(file_name + '/train.txt')
    train_model(training_data, feature_dic,
                                        log_name + '_' + time.strftime('%Y-%m-%d %H-%M-%S',
                                        time.localtime(time.time())))
    # print('Overall ROC-AUC:', average_auc)
    # print('Overall PR-AUC', average_pr)
    # print('Overall F1:', average_f1)