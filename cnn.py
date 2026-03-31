import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

SEQ_LEN = 16
TRAIN_RATIO = 0.80

def reduce_wesad_classes(data, binary_classification=False, three_class_classification=False):
    """Reduce the number of classes in the WESAD dataset
        In preprocessing (basic_seg): if labels[idx] not in [1, 2, 3]
        8 classes in total:
            0: baseline -> Keep for binary/three-class classification
            1: stress -> Keep for binary/three-class classification
            2: amusement -> Keep for three-class classification / Transform to 0 for binary classification
            3: meditation1
            4: rest
            5: meditation2
            6: ??
            7: ??
    """
    # TODO: Figure out which class to keep/transform for binary classification
    if not binary_classification and not three_class_classification:
        return data

    # in case of three-class classification, keep the first three classes
    selected_classes = sorted(data['label'].unique())[:3]
    data = data[data['label'].isin(selected_classes)]

    # in case of binary classification, transform amusement to baseline
    if binary_classification:
        data = data.replace({'label': 2}, {'label': 0}) # transform amusement to baseline
    return data

def create_windows(x, y, seq_len):
    X_windows = []
    y_labels = []
    step = seq_len
    for start in range(0, len(x) - seq_len + 1, step):
        end = start + seq_len
        window_x = x[start:end]
        window_y = y[start:end]

        X_windows.append(window_x)
        values, counts = np.unique(window_y, return_counts=True)
        y_labels.append(values[np.argmax(counts)])

    X_windows = np.array(X_windows)
    y_labels = np.array(y_labels)

    return X_windows, y_labels

def split_data(df_sub, train_ratio):

    split = int(len(df_sub) * train_ratio)

    train_df = df_sub.iloc[:split]
    test_df = df_sub.iloc[split:]
    
    return train_df, test_df

def normalize_data(X_train, X_test):
    n_train, seq_len, n_feat = X_train.shape
    n_test = X_test.shape[0]

    X_train_2d = X_train.reshape(-1, n_feat)
    X_test_2d  = X_test.reshape(-1, n_feat)

    scaler = StandardScaler()
    X_train_2d = scaler.fit_transform(X_train_2d)

    X_test_2d = scaler.transform(X_test_2d)

    X_train = X_train_2d.reshape(n_train, seq_len, n_feat)
    X_test  = X_test_2d.reshape(n_test, seq_len, n_feat)

    return X_train, X_test, scaler

def load_data():
    print("Loading data...")
    raw_data = pd.read_csv('./wesad_extracted.csv')

    print("Reducing classes to 3...")
    raw_data = reduce_wesad_classes(raw_data, False, True)
    
    print('Selecting chest signals...')
    raw_data = raw_data.drop(columns=[column for column in raw_data.columns if 'wrist' in column])
    
    print('Downsampling data...')
    raw_data = raw_data.iloc[::4].reset_index(drop=True)
    print(f'downsampled shape: ', raw_data.shape)
    subjects = raw_data['subject'].unique()
    feature_cols = ['ax', 'ay', 'az', 'emg', 'temp', 'eda', 'ecg', 'resp']
    label_col = 'label'

    train_X = []
    train_Y = []
    test_X = []
    test_Y = []
    for sub in subjects:
        sub_data = raw_data[raw_data['subject'] == sub].reset_index(drop=True)
        # παίρνουμε το 80% από κάθε subject για train και 20% για test
        train_data, test_data = split_data(sub_data, train_ratio=TRAIN_RATIO)

        train_x = train_data[feature_cols].values
        train_y = train_data[label_col].values

        test_x = test_data[feature_cols].values
        test_y = test_data[label_col].values

        if(len(train_x) != len(train_y)):
            print('X data are not equal with Y data')

        # χωρίζουμε σε παράθυρα το train και το test
        train_x_w, train_y_w = create_windows(train_x, train_y, seq_len=SEQ_LEN)
        test_x_w, test_y_w = create_windows(test_x, test_y, seq_len=SEQ_LEN)
        
        # προσθέτουμε τα καινούργια παράθυρα στην λίστα με τα παράθυρα από κάθε subject
        train_X.append(train_x_w)
        train_Y.append(train_y_w)
        test_X.append(test_x_w)
        test_Y.append(test_y_w)
    # κανονικοποιούμε τα δεδομένα μας
    train_X = np.concatenate(train_X)
    train_y = np.concatenate(train_Y)
    test_X = np.concatenate(test_X)
    test_y = np.concatenate(test_Y)

    print("train X shape:", train_X.shape)
    print("train y shape:", train_y.shape)
    print("test X shape:", test_X.shape)
    print("test y shape:", test_y.shape)
    train_X, test_X, scaler = normalize_data(train_X, test_X)

        

load_data()