# encoding: utf-8
"""
@author:    dawei gao
@contact:   david_gao@buaa.edu.cn

@version: 1.0
@license: Apache Licence
@file: dataset_mushroom.py
@time: 2018/10/23 7:48 PM

这一行开始写关于本文件的说明与解释
"""

import numpy as np


# 1: e, 0: p
# -1代表的是？
# 其他属性按照编号来处理
def attr(line):
    attrs_names = ['label',
                   'cap-shape',
                   'cap-surface',
                   'cap-color',
                   'bruises?',
                   'odor',
                   'gill-attachment',
                   'gill-spacing',
                   'gill-size',
                   'gill-color',
                   'stalk-shape',
                   'stalk-root',
                   'stalk-surface-above-ring',
                   'stalk-surface-below-ring',
                   'stalk-color-above-ring',
                   'stalk-color-below-ring',
                   'veil-type',
                   'veil-color',
                   'ring-number',
                   'ring-type',
                   'spore-print-color',
                   'population',
                   'habitat']
    embedding_dir = {
        'label': ['p', 'e'],
        'cap-shape': ['b', 'c', 'x', 'f', 'k', 's'],
        'cap-surface': ['f', 'g', 'y', 's'],
        'cap-color': ['n', 'b', 'c', 'g', 'r', 'p', 'u', 'e', 'w', 'y'],
        'bruises?': ['t', 'f'],
        'odor': ['a', 'l', 'c', 'y', 'f', 'm', 'n', 'p', 's'],
        'gill-attachment': ['a', 'd', 'f', 'n'],
        'gill-spacing': ['c', 'w', 'd'],
        'gill-size': ['b', 'n'],
        'gill-color': ['k', 'n', 'b', 'h', 'g', 'r', 'o', 'p', 'u', 'e', 'w', 'y'],
        'stalk-shape': ['e', 't'],
        'stalk-root': ['b', 'c', 'u', 'e', 'z', 'r'],
        'stalk-surface-above-ring': ['f', 'y', 'k', 's'],
        'stalk-surface-below-ring': ['f', 'y', 'k', 's'],
        'stalk-color-above-ring': ['n', 'b', 'c', 'g', 'o', 'p', 'e', 'w', 'y'],
        'stalk-color-below-ring': ['n', 'b', 'c', 'g', 'o', 'p', 'e', 'w', 'y'],
        'veil-type': ['p', 'u'],
        'veil-color': ['n', 'o', 'w', 'y'],
        'ring-number': ['n', 'o', 't'],
        'ring-type': ['c', 'e', 'f', 'l', 'n', 'p', 's', 'z'],
        'spore-print-color': ['k', 'n', 'b', 'h', 'r', 'o', 'u', 'w', 'y'],
        'population': ['a', 'c', 'n', 's', 'v', 'y'],
        'habitat': ['g', 'l', 'm', 'p', 'u', 'w', 'd']
    }
    attrs = line.replace('\n', '').split(',')
    attrs_res = list()
    count = 0
    print attrs
    for key in attrs_names:
        if count != 11:
            value = embedding_dir[key]
            item = [value.index(attrs[count])]
            attrs_res = attrs_res + item
        count += 1
    return str(attrs_res).replace('[', '').replace(']', '').replace(', ', '\t') + '\n'


def run():
    file_e = open('./datasets/mushroom_enlarging', 'w')
    file_t = open('./datasets/mushroom_tapering', 'w')

    file = open('./datasets/mushroom')
    for line in file:
        if line.split(',')[10] == 'e':
            file_e.write(attr(line))
        else:
            file_t.write(attr(line))


if __name__ == '__main__':
    run()
