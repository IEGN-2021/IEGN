#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 17:09:50 2018

@author: c503
"""

import numpy as np
import re
import itertools
from collections import Counter,defaultdict
import pandas as pd
import os.path as op
from scipy.sparse import lil_matrix

import torch
import csv
import math
import os
import pickle
from graph_data import *
import argparse

def load_pickle(name):
    with open(name, 'rb') as f:
        return pickle.load(f, encoding='latin1')


def save_pickle(obj, name, protocol=3):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, protocol=protocol)



def data_index_shift(lists, increase_by=2):
    """
    Increase the item index to contain the pad_index
    :param lists:
    :param increase_by:
    :return:
    """
    for seq in lists:
        for i, item_id in enumerate(seq):
            seq[i] = item_id + increase_by

    return lists


def time_slice(lists):
    """
    def timeSlice(user_timeset):
    time_min = min(user_timeset)
    time_map = dict()
    for time in user_timeset:
        time_map[time] = int(round(float(time-time_min)))
    return time_map
    """
    user_time = set()
    for seq in lists:
        for i, time_records in enumerate(seq):
            user_time.add(time_records)

    time_min = min(user_time)


    for seq in lists:

        time_diff = set()
        for i in range(len(seq) - 1):
           if seq[i + 1] - seq[i] != 0:
                time_diff.add(seq[i + 1] - seq[i])
        if len(time_diff) == 0:
             time_scale = 1
        else:
           time_scale = min(time_diff)

        for j, time_records in enumerate(seq):
            #print("seq[j]:", seq[j])
            seq[j] = int(round(float(time_records - time_min)))

    for seq in lists:

        seq_min = min(seq)
        for j, time_records in enumerate(seq):
            if j ==0:
                seq[j] = int(round((seq[j] - seq_min) / time_scale) + 1)
            else:
                seq[j] = int(round((seq[j] - seq_min-seq[j-1]) / time_scale) + 1)

    return lists


def split_data_sequentially(user_records, time_records, test_radio=0.2):
    train_set = []
    test_set = []
    train_time_val = []
    test_time = []
    for item_list in user_records:
        len_list = len(item_list)
        num_test_samples = int(math.ceil(len_list * test_radio))
        train_sample = []
        test_sample = []
        for i in range(len_list - num_test_samples, len_list):
            test_sample.append(item_list[i])

        for place in item_list:
            if place not in set(test_sample):
                train_sample.append(place)

        train_set.append(train_sample)
        test_set.append(test_sample)

    for time_list in time_records:
        len_time_list = len(time_list)
        num_test_time_samples = int(math.ceil(len_time_list * test_radio))
        train_time_sample = []
        test_time_sample = []
        for i in range(len_time_list - num_test_time_samples, len_time_list):
            test_time_sample.append(time_list[i])

        for j in range(0,len_time_list - num_test_time_samples):
            train_time_sample.append(time_list[j])

        train_time_val.append(train_time_sample)
        test_time.append(test_time_sample)

    return train_set, train_time_val, test_set, test_time


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    TPS_DIR = '../data/music'

    parser.add_argument('--data_path', type=str, default='../data/music/')
    parser.add_argument('--user_data', type=str, default='../data/music/music.csv')
    parser.add_argument('--adj_type', type=str, default='plain')
    parser.add_argument('--dir_path', type=str, default='../pro_data')

    # add parameters
    parser.add_argument('--user_record_file', type=str, default='/CDs_item_sequences.pkl')
    parser.add_argument('--user_mapping_file', type=str, default='/CDs_user_mapping.pkl')
    parser.add_argument('--item_mapping_file', type=str, default='/CDs_item_mapping.pkl')
    parser.add_argument('--time_record_file', type=str, default='/CDs_time_sequences.pkl')

    config = parser.parse_args()

    index_shift = 1
    num_users = 1
    num_items = 1

    user_records = None
    user_mapping = None
    item_mapping = None

    user_records = load_pickle(config.dir_path + config.user_record_file)
    user_mapping = load_pickle(config.dir_path + config.user_mapping_file)
    item_mapping = load_pickle(config.dir_path + config.item_mapping_file)
    time_records = load_pickle(config.dir_path + config.time_record_file)
    print("len(user_mapping):",len(user_mapping))

    num_users = len(user_mapping)
    num_items = len(item_mapping)

    time_records = time_slice(time_records)
    user_records = data_index_shift(user_records, increase_by=index_shift)
    time_records = data_index_shift(time_records,increase_by=index_shift )
    num_items = num_items + index_shift

    data_generator = Data(path=config.data_path, user_records=user_records, user_num=num_users, item_num=num_items)

    plain_adj, norm_adj, mean_adj, pre_adj = data_generator.get_adj_mat()

    # split dataset
    train_val_set, train_time_val, test_set, test_time = split_data_sequentially(user_records, time_records,
                                                                                 test_radio=0.2)

    #return train_val_set, train_time_val, test_set, test_time, num_users, num_items, plain_adj, norm_adj, mean_adj, pre_adj

    
    np.random.seed(2019)

    para={}
    para['user_num']=num_users
    para['item_num']=num_items
    para['user_train'] = train_val_set
    para['user_test'] = test_set

    para['train_time_val'] = train_time_val
    para['test_time'] = test_time

    if config.adj_type == 'plain':
        para['plain_adj'] = plain_adj
        print('use the plain adjacency matrix')

    output = open(os.path.join(TPS_DIR, 'music_test.para'), 'wb')

    pickle.dump(para, output)
