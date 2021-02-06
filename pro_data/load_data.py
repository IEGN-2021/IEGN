#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 17:20:22 2018

@author: c503
"""

import os
import json
import pandas as pd
import pickle
import numpy as np
from copy import deepcopy


import json
import gzip
dir_path = '../data/music/'

rating_file = 'ratings_Beauty.csv'
review_file = 'reviews_Beauty_5.json.gz'

def read_user_rating_records():
    col_names = ['user_id', 'item_id', 'rating', 'timestamp']
    data_records = pd.read_csv(dir_path + rating_file, sep=',', names=col_names, engine='python')
    return data_records

data_records = read_user_rating_records()
data_records.head()
data_records.iloc[[1, 10, 20]]

data_records.loc[data_records.rating < 4, 'rating'] = 0
data_records.loc[data_records.rating >= 4, 'rating'] = 1
data_records = data_records[data_records.rating > 0]


def remove_infrequent_items(data, min_counts=5):
    df = deepcopy(data)
    counts = df['item_id'].value_counts()
    df = df[df["item_id"].isin(counts[counts >= min_counts].index)]

    print("items with < {} interactoins are removed".format(min_counts))
    # print(df.describe())
    return df

def remove_infrequent_users_max(data, mam_counts=10):
    df = deepcopy(data)
    counts = df['item_id'].value_counts()
    df = df[df["item_id"].isin(counts[counts<= mam_counts].index)]

    print("items with > {} interactoins are removed".format(mam_counts))
    # print(df.describe())
    return df

def remove_infrequent_users(data, min_counts=8):
    df = deepcopy(data)
    counts = df['user_id'].value_counts()
    df = df[df["user_id"].isin(counts[counts >= min_counts].index)]

    print("users with < {} interactoins are removed".format(min_counts))
    # print(df.describe())
    return df

filtered_data = remove_infrequent_users(data_records, 5)

filtered_data = remove_infrequent_items(filtered_data, 8)

# read item's reviews
item_list = filtered_data['item_id'].unique()
item_set = set(item_list)

time_list = filtered_data['timestamp'].unique()
time_list = set(time_list)

print(item_list[:10])




def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield json.loads(l.decode())
        # yield json.dumps(eval(l))

review_dict = dict()  # [review_id] = review_text
review_helpful = dict()
for l in parse(dir_path + review_file):
    if l['asin'] in item_set:
        if l['asin'] in review_dict:
            if l['helpful'][0] / float(l['helpful'][1] + 0.01) > review_helpful[l['asin']] and len(l['reviewText']) > 10:
                review_dict[l['asin']] = l['reviewText']
                review_helpful[l['asin']] = l['helpful'][0] / float(l['helpful'][1] + 0.01)
        else:
            if len(l['reviewText']) > 10:
                review_dict[l['asin']] = l['reviewText']
                review_helpful[l['asin']] = l['helpful'][0] / float(l['helpful'][1] + 0.01)

# print review_dict['1300966947']

# delete items without reviews
item_without_review = []
for item_id in item_list:
    if item_id not in review_dict:
        item_without_review.append(item_id)

#print(item_without_review)

for item_id in item_without_review:
    filtered_data = filtered_data[filtered_data['item_id'] != item_id]

item_list = filtered_data['item_id'].unique()
print(len(item_list))

for item_id, review in review_dict.items():
    if len(review) < 5:
        print(item_id)
# print review_dict['B002IUAUI2']



# convert records to sequential data per user
def convert_data(data):
    # for each user, sort by timestamps
    df = deepcopy(data)
    df_ordered = df.sort_values(['timestamp'], ascending=True)
    data = df_ordered.groupby('user_id')['item_id'].apply(list)

    print("succressfully created sequencial data! head:", data.head(5))
    unique_data = df_ordered.groupby('user_id')['item_id'].nunique()

    data = data[unique_data[unique_data >= 10].index]

    data_time = df_ordered.groupby('user_id')['timestamp'].apply(list)
    print("succressfully created sequencial tiem_data! head:", data.head(5))
    data_time = data_time[unique_data[unique_data >= 10].index]


    return data, data_time

seq_data, seq_time_data = convert_data(filtered_data)


user_item_dict = seq_data.to_dict()
user_time_dict = seq_time_data.to_dict()

user_mapping = []
item_set = set()
rating_count = 0
for user_id, item_list in seq_data.iteritems():
    user_mapping.append(user_id)
    rating_count +=len(item_list)
    for item_id in item_list:
        item_set.add(item_id)
item_mapping = list(item_set)

print("len(user_mapping):",len(user_mapping), len(item_mapping),rating_count)

def generate_inverse_mapping(data_list):
    inverse_mapping = dict()
    for inner_id, true_id in enumerate(data_list):
        inverse_mapping[true_id] = inner_id
    return inverse_mapping

def convert_to_inner_index(user_records, user_mapping, item_mapping):
    inner_user_records = []
    user_inverse_mapping = generate_inverse_mapping(user_mapping)
    item_inverse_mapping = generate_inverse_mapping(item_mapping)

    for user_id in range(len(user_mapping)):
        real_user_id = user_mapping[user_id]
        item_list = list(user_records[real_user_id])
        for index, real_item_id in enumerate(item_list):
            item_list[index] = item_inverse_mapping[real_item_id]
        inner_user_records.append(item_list)

    return inner_user_records, user_inverse_mapping, item_inverse_mapping

def convert_to_inner_index_time(user_time_dict, user_mapping):
    inner_user_time_records = []

    for user_id in range(len(user_mapping)):
        real_user_id = user_mapping[user_id]
        time_list = list(user_time_dict[real_user_id])
        inner_user_time_records.append(time_list)

    return inner_user_time_records

inner_data_records, user_inverse_mapping, item_inverse_mapping = convert_to_inner_index(user_item_dict, user_mapping, item_mapping)
inner_data_time = convert_to_inner_index_time(user_time_dict,user_mapping)

print(inner_data_records[:5])
print(inner_data_time[:5])

def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f)

save_obj(inner_data_records, 'CDs_item_sequences')
save_obj(user_inverse_mapping, 'CDs_user_mapping')
save_obj(item_inverse_mapping, 'CDs_item_mapping')
save_obj(inner_data_time, 'CDs_time_sequences')
print("CDs_user_mapping:", len(user_inverse_mapping),len(inner_data_time))
