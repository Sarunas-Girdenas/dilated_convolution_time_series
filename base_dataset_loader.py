from abc import ABC, abstractmethod
import pandas as pd
from numpy import array_split, concatenate
from random import shuffle
from itertools import product
from collections import Counter

from sklearn.preprocessing import MinMaxScaler

class BaseDataLoader(ABC):
    """
    Base class for various data loaders
    We use to feed data to various Deep Learning models
    """

    def _stack_sequences(self,
                         input_data: pd.core.frame.DataFrame,
                         input_labels: pd.core.frame.DataFrame,
                         seq_lenght: int,
                         feature_names: list,
                         pairs: tuple):
        """
        Stack sequences for training & testing
        Inputs:
        =======
        input_data (pd.core.frame.DataFrame): raw data to be stacked
        input_labels (pd.core.frame.DataFrame): labels
        seq_length (int): length of the sequence
        num_features (int): number of features to be stacked
        pairs (tuple): pairs
        """

        num_features = len(feature_names)

        indices = list(range(int(len(input_data)/len(pairs))-seq_lenght))

        indices_split = array_split(indices, len(indices)/seq_lenght)
        shuffle(indices_split)

        # convert to list because we want to use indices twice
        sequence_indices = list(product(indices_split, pairs))

        stacked_labels = []

        stacked = None

        # calculate most common length of indices
        lenghts = [len(i[0]) for i in sequence_indices]
        most_common_length = Counter(lenghts).most_common()[0][0]

        self.sequence_length = most_common_length

        # stack together
        for seq in sequence_indices:
            if len(seq[0]) == most_common_length:
                if stacked is None:
                    stacked = input_data.query(
                        f"symbol == '{seq[1]}'")[feature_names].iloc[seq[0]].values.reshape(
                            1, num_features, most_common_length)
                    # label index
                    temp_idx = input_data.query(f"symbol == '{seq[1]}'").iloc[seq[0][-1]].name
                    # label value
                    label_value = input_labels.query(f"index == '{temp_idx}'")[f"{seq[1]}_label"].values[0]
                    stacked_labels.append(label_value)
                    
                if stacked is not None:
                    temp_len = len(input_data.query(
                        f"symbol == '{seq[1]}'")[feature_names].iloc[seq[0]])
                    temp_stacked = input_data.query(
                        f"symbol == '{seq[1]}'")[feature_names].iloc[seq[0]].values.reshape(
                            1, num_features, temp_len)
                    stacked = concatenate([stacked, temp_stacked])
                    # label index
                    temp_idx = input_data.query(f"symbol == '{seq[1]}'").iloc[seq[0][-1]].name
                    # label value
                    label_value = input_labels.query(f"index == '{temp_idx}'")[f"{seq[1]}_label"].values[0]
                    stacked_labels.append(label_value)
 
            else:
                pass
        
        return stacked, stacked_labels
    
    @property
    def actual_sequence_length(self):
        """
        Returns actual lenght of the sequence
        that was used.
        It might be be different than requested.
        This happens due to the way index is split
        into sub-intervals.
        """

        return self.sequence_length

    def _compute_labels(self,
                        data: pd.core.frame.DataFrame,
                        pairs: tuple,
                        columns: tuple,):
        """
        Given the data, compute labels
        Label - is the next bid price higher than the current spread?
        """
        
        for idx, cols in enumerate(columns):
            if idx == 0:
                prices_pivoted = pd.pivot_table(data, index=data.index, columns='symbol', values=cols)
                prices_pivoted.rename(
                    columns=dict(
                        zip(prices_pivoted.columns, [f"{i}_{cols}" for i in prices_pivoted.columns])), inplace=True)
                labels = prices_pivoted
            else:
                prices_pivoted = pd.pivot_table(data, index=data.index, columns='symbol', values=cols)
                prices_pivoted.rename(
                    columns=dict(
                        zip(prices_pivoted.columns, [f"{i}_{cols}" for i in prices_pivoted.columns])), inplace=True)
                labels = pd.merge(labels, prices_pivoted, left_index=True, right_index=True)
        
        labels_columns = []
        
        for pair in pairs:
            # 1. Shift bidPrice up by one
            labels[f"{pair}_bidPrice_shifted"] = labels[f"{pair}_bidPrice"].shift(-1)
            
            # 2. Calculate difference between current ask and future bid
            labels['bid_-1_ask'] = labels[f"{pair}_bidPrice_shifted"].values - labels[f"{pair}_askPrice"].values
            
            # 3. Convert to label
            labels[f"{pair}_label"] = labels['bid_-1_ask'].map(lambda x: 1 if x > 0 else 0)
            
            # 4. Drop used columns
            labels.drop(['bid_-1_ask', f"{pair}_bidPrice_shifted"], axis=1, inplace=True)
            
            # 5. Select only labels columns
            labels_columns.append(f"{pair}_label")
        
        return labels[labels_columns]