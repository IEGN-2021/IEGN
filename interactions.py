"""
Classes describing datasets of user-item interactions. Instances of these
are returned by dataset-fetching and dataset-processing functions.
"""

import numpy as np

import scipy.sparse as sp


class Interactions(object):
    """
    Interactions object. Contains (at a minimum) pair of user-item
    interactions. This is designed only for implicit feedback scenarios.

    Parameters
    ----------

    file_path: file contains (user,item,rating) triplets
    user_map: dict of user mapping
    item_map: dict of item mapping
    """

    def __init__(self, user_item_sequence, train_time, num_users, num_items):
        user_ids, item_ids, time_item = [], [], []
        for uid, item_seq in enumerate(user_item_sequence):
            for iid in item_seq:
                user_ids.append(uid)
                item_ids.append(iid)

        user_ids = np.asarray(user_ids)
        item_ids = np.asarray(item_ids)

        self.num_users = num_users
        self.num_items = num_items

        self.user_ids = user_ids
        self.item_ids = item_ids

        #add times
        for uid, time_seq in enumerate(train_time):
            for time_id in time_seq:
                time_item.append(time_id)

        time_item = np.asarray(time_item)
        self.time_item = time_item

        self.time_squeueces = None

        self.sequences = None
        self.test_sequences = None

    def __len__(self):

        return len(self.user_ids)

    def tocoo(self):
        """
        Transform to a scipy.sparse COO matrix.
        """

        row = self.user_ids
        col = self.item_ids
        data = np.ones(len(self))

        return sp.coo_matrix((data, (row, col)),
                             shape=(self.num_users, self.num_items))

    def tocsr(self):
        """
        Transform to a scipy.sparse CSR matrix.
        """

        return self.tocoo().tocsr()

    def to_sequence(self, sequence_length=5, target_length=1):
        """
        Transform to sequence form.

        Valid subsequences of users' interactions are returned. For
        example, if a user interacted with items [1, 2, 3, 4, 5, 6, 7, 8, 9], the
        returned interactions matrix at sequence length 5 and target length 3
        will be be given by:

        sequences:

           [[1, 2, 3, 4, 5],
            [2, 3, 4, 5, 6],
            [3, 4, 5, 6, 7]]

        targets:

           [[6, 7],
            [7, 8],
            [8, 9]]

        sequence for test (the last 'sequence_length' items of each user's sequence):

        [[5, 6, 7, 8, 9]]

        Parameters
        ----------

        sequence_length: int
            Sequence length. Subsequences shorter than this
            will be left-padded with zeros.
        target_length: int
            Sequence target length.
        """

        # # change the item index start from 1 as 0 is used for padding in sequences
        # for k, v in self.item_map.items():
        #     self.item_map[k] = v + 1
        # self.item_ids = self.item_ids + 1
        # self.num_items += 1

        max_sequence_length = sequence_length + target_length

        # Sort first by user id
        sort_indices = np.lexsort((self.user_ids,))

        user_ids = self.user_ids[sort_indices]
        item_ids = self.item_ids[sort_indices]

        # 修改 time_train
        time_train = self.time_item[sort_indices]

        user_ids, indices, counts = np.unique(user_ids,
                                              return_index=True,
                                              return_counts=True)

        num_subsequences = sum([c - max_sequence_length + 1 if c >= max_sequence_length else 1 for c in counts])

        sequences = np.zeros((num_subsequences, sequence_length),
                             dtype=np.int64)
        sequences_targets = np.zeros((num_subsequences, target_length),
                                     dtype=np.int64)
        sequence_users = np.empty(num_subsequences,
                                  dtype=np.int64)

        test_sequences = np.zeros((self.num_users, sequence_length),
                                  dtype=np.int64)
        test_users = np.empty(self.num_users,
                              dtype=np.int64)

        # 修改
        sequences_time = np.zeros((num_subsequences, sequence_length),
                                  dtype=np.int64)
        sequences_time_targets = np.zeros((num_subsequences, target_length),
                                          dtype=np.int64)
        test_time_sequences = np.zeros((self.num_users, sequence_length),
                                       dtype=np.int64)

        _uid = None
        for i, (uid, item_seq, time_seq) in enumerate(_generate_sequences(user_ids,
                                                                          item_ids, time_train,
                                                                          indices,
                                                                          max_sequence_length)):
            if uid != _uid:
                test_sequences[uid][:] = item_seq[-sequence_length:]    #？？？？？？？确定是这样
                test_time_sequences[uid][:] = time_seq[-sequence_length:]
                test_users[uid] = uid
                _uid = uid
            sequences_targets[i][:] = item_seq[-target_length:]
            sequences_time_targets[i][:] = time_seq[-target_length:]
            sequences[i][:] = item_seq[:sequence_length]
            sequences_time[i][:] = time_seq[:sequence_length]
            sequence_users[i] = uid



        self.sequences = SequenceInteractions(sequence_users, sequences, sequences_time, sequences_targets,
                                              sequences_time_targets)
        self.test_sequences = SequenceInteractions(test_users, test_sequences, test_time_sequences)




class SequenceInteractions(object):
    """
    Interactions encoded as a sequence matrix.

    Parameters
    ----------
    user_ids: np.array
        sequence users
    sequences: np.array
        The interactions sequence matrix, as produced by
        :func:`~Interactions.to_sequence`
    targets: np.array
        sequence targets
    """

    def __init__(self,
                 user_ids,
                 sequences, sequences_time,
                 targets=None, targets_time=None):
        self.user_ids = user_ids
        self.sequences = sequences
        self.sequences_time = sequences_time
        self.targets = targets
        self.targets_time = targets_time

        self.L = sequences.shape[1]
        self.T = None
        if np.any(targets):
            self.T = targets.shape[1]


def _sliding_window(tensor, tensor_t, window_size, step_size=1):
    if len(tensor) - window_size >= 0 & len(tensor_t) - window_size >= 0:
        for i in range(len(tensor), 0, -step_size):
            if i - window_size >= 0:
                yield tensor[i - window_size:i], tensor_t[i - window_size:i]
            else:
                break
    else:
        num_paddings = window_size - len(tensor)
        num_paddings_t = window_size - len(tensor_t)
        # Pad sequence with 0s if it is shorter than windows size.
        yield np.pad(tensor, (num_paddings, 0), 'constant'), np.pad(tensor_t, (num_paddings_t, 0), 'constant')


def _generate_sequences(user_ids, item_ids, time_train,
                        indices,
                        max_sequence_length):
    for i in range(len(indices)):

        start_idx = indices[i]

        if i >= len(indices) - 1:
            stop_idx = None
        else:
            stop_idx = indices[i + 1]

        for seq, seq_time in _sliding_window(item_ids[start_idx:stop_idx], time_train[start_idx:stop_idx],
                                             max_sequence_length):
            yield (user_ids[i], seq, seq_time)
