from ast import literal_eval
from tcn_data_loader import TcnDataLoader
from tcn_model import DilatedNet
import configparser
from torch import nn
import torch
from torch.utils import data
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
import optuna

NUM_FEATURES = 5
FEATURES = ['askPrice', 'bidPrice', 'bidQty/askQty', 'volume', 'priceChangePercent']
TRAIN_SET_SIZE = 0.85
KERNEL_SIZE = 2
GRAD_CLIPPING_VAL = 1
CONFIG_LOCATION = '../ml_models_for_airflow/dbs3_config.ini'
EARLY_STOPPING = 0.001
EARLY_STOPPING_EPOCHS = 3

# Read in config
config = configparser.ConfigParser()
config.read(CONFIG_LOCATION)

pairs_mapping = literal_eval(config['MODEL']['pairs_mapping'])
pairs = tuple(pairs_mapping.values())

def define_model(trial):
    """
    Define model structure
    """

    dilation = trial.suggest_int('dilation', 1, 8)
    depth = trial.suggest_int('depth', 1, 10)

    seq_length = trial.suggest_int('seq_length', 50, 200)
    out_channels = trial.suggest_int('out_channels', 2, 164)

    return out_channels, dilation, depth, seq_length

def objective(trial):
    """
    Prepare training data (lenght of sequences)
    and train the model.
    """

    out_channels, dilation, depth, seq_length = define_model(trial)

    full_data_set = TcnDataLoader(
        config_location=CONFIG_LOCATION,
        pairs=pairs,
        seq_lenght=seq_length,
        features=FEATURES,
        local_path_book='book_data_tcn.csv',
        local_path_volume='volume_data_tcn.csv' 
        )
    
    model = DilatedNet(
        num_features=NUM_FEATURES,
        out_channels=out_channels,
        dilation=dilation,
        depth=depth,
        seq_length=full_data_set.actual_sequence_length,
        kernel_size=KERNEL_SIZE
    )
    model.apply(model.init_weights)

    train_set_size = int(len(full_data_set)*TRAIN_SET_SIZE)
    test_set_size = len(full_data_set) - train_set_size

    trainset, testset = data.random_split(full_data_set,
                                        [train_set_size, test_set_size]
                                    )

    batch_size = trial.suggest_int('batch_size', 16, 300)

    train_generator = data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4)
    test_generator = data.DataLoader(
        testset,
        batch_size=len(testset),
        shuffle=True,
        num_workers=4)

    num_epochs = trial.suggest_int('num_epochs', 3, 60)

    # Generate the optimizers.
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    lr = trial.suggest_loguniform("learning_rate", 1e-5, 1e-1)
    optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=lr)
    learning_rate_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda i: min(i / (10.0 / batch_size), 1.0))

    criterion = torch.nn.BCELoss()
    train_auc = []
    test_auc = []

    for ep in range(num_epochs):
        model.train()
        epoch_loss = 0
        temp_train_auc = 0
        
        for train_x, train_y in train_generator:
            
            predictions = model(train_x)
            loss = criterion(predictions, train_y.view(-1, 1))
            epoch_loss += loss.item()
            try:
                temp_train_auc += roc_auc_score(
                    train_y.numpy(), predictions.detach().numpy())
            except ValueError:
                temp_train_auc += 0.5
            
            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIPPING_VAL)

            optimizer.step()
            learning_rate_scheduler.step()
        
        train_auc.append(temp_train_auc/len(train_generator))
        
        with torch.no_grad():
            model.eval()
            temp_test_auc = 0
            for test_x, test_y in test_generator:
                predictions = model(test_x)
                temp_test_auc += roc_auc_score(
                    test_y.numpy(), predictions.numpy())

        test_auc.append(temp_test_auc/len(test_generator))

        # Early Stopping
        if len(test_auc) > EARLY_STOPPING_EPOCHS:
            if max([x[1]-x[0] for x in zip(test_auc[1:], test_auc[:-1])][-EARLY_STOPPING_EPOCHS:]) <= EARLY_STOPPING:
                print('Training Stopped by Early Stopping!')
                break

        if ep % 2 == 0: print('test auc:', test_auc[-1], ' epoch:', ep)
    
    return test_auc[-1]

if __name__ == '__main__':
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100, timeout=12000, n_jobs=4)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))