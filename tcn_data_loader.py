import configparser
from sqlalchemy import create_engine
import torch
from torch.utils import data
import pandas as pd
from numpy import array_split, concatenate
from random import shuffle
from itertools import product
from sklearn.preprocessing import MinMaxScaler

from ml_model_files.base_dataset_loader import BaseDataLoader

class TcnDataLoader(data.Dataset, BaseDataLoader):
    """
    Loads sequence for TCN models
    """

    def __init__(self, config_location: str,
                 pairs: tuple, seq_lenght: int,
                 features: list,
                 local_path_book: str=None,
                 local_path_volume: str=None):
        """
        Load config file with DB connections
        Inputs:
        =======
        config_location (str): location of config file
        pairs (tuple): currency pairs
        seq_length (int): lenght of sequence to use for training
        features (list): features to use
        """

        super(TcnDataLoader, self).__init__()

        config = configparser.ConfigParser()
        config.read(config_location)

        db_engine = create_engine(config['AIRFLOW']['postgres_conn'])

        if local_path_book is None:
            book_data = pd.read_sql(
                f"select * from current_book where symbol in {tuple(pairs)}",
                db_engine
            )
        if local_path_volume is None:
            volume_data = pd.read_sql(
                f"""select timestamp, symbol, volume, "priceChangePercent" from current_market_price_24_hours where symbol in {tuple(pairs)}""",
                db_engine
            )

        db_engine.dispose()
        
        if local_path_book is not None:
            book_data = pd.read_csv(local_path_book)
        if local_path_volume is not None:
            volume_data = pd.read_csv(local_path_volume)

        # prepare book data
        book_data['askPrice'] = book_data['askPrice'].astype(float)
        book_data['bidPrice'] = book_data['bidPrice'].astype(float)
        book_data['askQty'] = book_data['askQty'].astype(float)
        book_data['bidQty'] = book_data['bidQty'].astype(float)
        book_data.index = pd.to_datetime(book_data['timestamp'])
        book_data['formatted_index'] = book_data.index.map(
            lambda x: x.strftime(format='%Y-%m-%d %H:%M')
            )
        book_data.drop('timestamp', axis=1, inplace=True)

        book_data['bidQty/askQty'] = book_data['bidQty'] / (book_data['bidQty'] + book_data['askQty'])
        book_data.drop(['bidQty', 'askQty'], inplace=True, axis=1)

        # prepare volume data
        volume_data['volume'] = volume_data['volume'].astype(float)
        volume_data['priceChangePercent'] = volume_data['priceChangePercent'].astype(float)
        volume_data.index = pd.to_datetime(volume_data['timestamp'])
        volume_data['formatted_index'] = volume_data.index.map(
            lambda x: x.strftime(format='%Y-%m-%d %H:%M')
            )
        volume_data.drop('timestamp', axis=1, inplace=True)

        data = pd.merge(book_data, volume_data, on=['formatted_index', 'symbol'])
        data.drop_duplicates(subset='formatted_index', keep='last')
        data.index = pd.to_datetime(data['formatted_index'])
        # use only required columns
        features_data = data[features].copy()

        scaler = MinMaxScaler()
        transformed = scaler.fit_transform(features_data.values)
        features_data = pd.DataFrame(transformed, columns=features)

        self.labels = self._compute_labels(data=book_data, pairs=pairs, columns=('bidPrice', 'askPrice'))
        self.labels.index = pd.to_datetime(self.labels.index.map(
            lambda x: x.strftime(format='%Y-%m-%d %H:%M')
            )
        )
        
        # index to merge labels with data
        features_data.index = pd.to_datetime(data['formatted_index'])
        features_data['symbol'] = data['symbol'].values

        self.stacked_sequences, self.stacked_labels = self._stack_sequences(
            input_data=features_data,
            input_labels=self.labels,
            seq_lenght=seq_lenght,
            feature_names=features,
            pairs=pairs
        )

        return None
    
    def __len__(self):
        """
        Returns total number of samples in
        the dataset
        """

        return len(self.stacked_sequences)
    
    def __getitem__(self, index):
        """
        Generate samples of data
        """

        x = torch.tensor(self.stacked_sequences).float()[index]
        y = torch.tensor(self.stacked_labels).float()[index]

        return x, y