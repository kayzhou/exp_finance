# -*- coding: utf-8 -*-
__author__ = 'Kay'


def feature_selection(in_name, out_name, selected):
    in_file = open(in_name)
    out_file = open(out_name, 'w')
    for line in in_file:
        line = line.strip().split(' ')
        y = 'init'
        X = []
        for i, index_feature in enumerate(line):
            if i == 0:
                y = index_feature
            else:
                index, feature = index_feature.split(':')
                if int(index) + 1 in selected:
                    # print(index)
                    X.append(feature)
        out_file.write(y + ' ' + ' '.join([str(i) + ':' + str(xi) for i, xi in enumerate(X)]) + '\n')


if __name__ == '__main__':
    # feature_selection('../data/proportion/SVM_DATA/class3_GOAL0.txt', '../data/proportion/SVM_DATA/selected_class3_GOAL0.txt',
    #                   [2, 7])
    # feature_selection('../data/proportion/SVM_DATA/class3_GOAL1.txt', '../data/proportion/SVM_DATA/selected_class3_GOAL1.txt',
    #                   [3, 8, 13, 18, 23, 5, 10, 15, 20, 25, 12, 17])
    # feature_selection('../data/proportion/SVM_DATA/class3_GOAL2.txt', '../data/proportion/SVM_DATA/selected_class3_GOAL2.txt',
    #                   [3, 8, 13, 18, 4, 9, 14, 22])
    # feature_selection('../data/proportion/SVM_DATA/class3_GOAL3.txt', '../data/proportion/SVM_DATA/selected_class3_GOAL3.txt',
    #                   [4, 13, 18, 2])
    # feature_selection('../data/proportion/SVM_DATA/class3_GOAL4.txt', '../data/proportion/SVM_DATA/selected_class3_GOAL4_sadness_fear.txt',
    #                   [5, 10, 15, 20, 25, 4, 9, 14, 19, 24])
    # feature_selection('../data/proportion/SVM_DATA/class2_GOAL0.txt', '../data/proportion/SVM_DATA/selected_class2_GOAL0.txt',
    #                   [2, 7])
    # feature_selection('../data/proportion/SVM_DATA/class2_GOAL1.txt', '../data/proportion/SVM_DATA/selected_class2_GOAL1.txt',
    #                   [3, 8, 13, 18, 23, 5, 10, 15, 20, 25, 12, 17])

    # 测试数据
    feature_selection('../data/proportion/测试数据/20150917-20151207_比例_三分类_GOAL0.txt', '../data/proportion/测试数据/selected_class3_GOAL0.txt',
                      [2, 7])
    feature_selection('../data/proportion/测试数据/20150917-20151207_比例_三分类_GOAL1.txt', '../data/proportion/测试数据/selected_class3_GOAL1.txt',
                      [3, 8, 13, 18, 23, 5, 10, 15, 20, 25, 12, 17])
    feature_selection('../data/proportion/测试数据/20150917-20151207_比例_三分类_GOAL2.txt', '../data/proportion/测试数据/selected_class3_GOAL2.txt',
                      [3, 8, 13, 18, 4, 9, 14, 22])
    feature_selection('../data/proportion/测试数据/20150917-20151207_比例_三分类_GOAL3.txt', '../data/proportion/测试数据/selected_class3_GOAL3.txt',
                      [4, 13, 18, 2])
    feature_selection('../data/proportion/测试数据/20150917-20151207_比例_三分类_GOAL4.txt', '../data/proportion/测试数据/selected_class3_GOAL4.txt',
                      [5, 10, 15, 20, 25, 4, 9, 14, 19, 24])
    feature_selection('../data/proportion/测试数据/20150917-20151207_比例_二分类_GOAL0.txt', '../data/proportion/测试数据/selected_class2_GOAL0.txt',
                      [2, 7])
    feature_selection('../data/proportion/测试数据/20150917-20151207_比例_二分类_GOAL1.txt', '../data/proportion/测试数据/selected_class2_GOAL1.txt',
                      [3, 8, 13, 18, 23, 5, 10, 15, 20, 25, 12, 17])