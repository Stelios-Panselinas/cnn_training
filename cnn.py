import pandas as pd
import numpy as np
import os
import sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, GlobalAveragePooling1D, Input
from tensorflow.keras.optimizers import Adam

SEQ_LEN = 16
TRAIN_RATIO = 0.80
NUM_CLASSES = 3
FILTER1 = 3
FILTER2 = 6
KERNEL_SIZE = 2
POOL_SIZE = 2

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

def build_cnn(seq_len, num_features, num_classes=NUM_CLASSES):
    model = Sequential([
        Input(shape=(seq_len, num_features)),

        Conv1D(filters=FILTER1, kernel_size=KERNEL_SIZE, activation='relu', padding='same'),
        # BatchNormalization(),
        MaxPooling1D(pool_size=POOL_SIZE),

        Conv1D(filters=FILTER2, kernel_size=KERNEL_SIZE, activation='relu', padding='same'),
        # BatchNormalization(),
        # MaxPooling1D(pool_size=2),

        # BatchNormalization(),

        GlobalAveragePooling1D(),

        Dense(8, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

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
    feature_cols = ['ax', 'ay', 'az', 'emg', 'temp', 'eda', 'ecg', 'resp']
    label_col = 'label'

    return raw_data, feature_cols, label_col

def split_and_prepare_data(data):
    train_dfs = []
    test_dfs = []

    subjects = data['subject'].unique()

    for sub in subjects:
        sub_data = data[data['subject'] == sub].reset_index(drop=True)

        # split
        train_data, test_data = split_data(sub_data, train_ratio=TRAIN_RATIO)

        # extract numpy
        train_x = train_data[feature_cols].values
        train_y = train_data[label_col].values

        test_x = test_data[feature_cols].values
        test_y = test_data[label_col].values

        # windowing
        train_x_w, train_y_w = create_windows(train_x, train_y, seq_len=SEQ_LEN)
        test_x_w, test_y_w = create_windows(test_x, test_y, seq_len=SEQ_LEN)

        train_x_w, test_x_w, scaler = normalize_data(train_x_w, test_x_w)
        # create DataFrames for this subject
        train_df_sub = pd.DataFrame({
            'subject': sub,
            'X': list(train_x_w),   # store each window as array
            'y': train_y_w
        })

        test_df_sub = pd.DataFrame({
            'subject': sub,
            'X': list(test_x_w),
            'y': test_y_w
        })

        train_dfs.append(train_df_sub)
        test_dfs.append(test_df_sub)

    # concatenate all subjects
    train_df = pd.concat(train_dfs, ignore_index=True)
    test_df = pd.concat(test_dfs, ignore_index=True)

    return train_df, test_df

def train_model(train_df, test_df, option):
    subjects = train_df['subject'].unique()
    if(option == '0'):
        for sub in subjects:
            train_X = np.stack(train_df[train_df['subject'] == sub]['X'])
            train_Y = np.stack(train_df[train_df['subject'] == sub]['y'])

            test_X = np.stack(test_df[test_df['subject'] == sub]['X'])
            test_X = np.stack(test_df[test_df['subject'] == sub]['y'])

            model = build_cnn(seq_len=SEQ_LEN, num_features=8, num_classes=NUM_CLASSES)
            history = model.fit(
            train_X, train_Y,
            validation_data=(test_X, test_Y),
            epochs=10,
            batch_size=32,
            verbose=1
            )
    elif(option == '1'):    
        train_X = np.stack(train_df['X'].values)
        train_Y = train_df['y'].values

        test_X = np.stack(test_df['X'].values)
        test_Y = test_df['y'].values

        
        #.........
        model = build_cnn(seq_len=SEQ_LEN, num_features=8, num_classes=NUM_CLASSES)
        history = model.fit(
        train_X, train_Y,
        validation_data=(test_X, test_Y),
        epochs=10,
        batch_size=32,
        verbose=1
        )
    # test_loss, test_acc = model.evaluate(test_X, test_Y, verbose=0)
    # print("Test accuracy:", test_acc)
    return model
        
    

if __name__ == "__main__":  
    option = sys.argv[1]     
    print(option)
    data, feature_cols, label_col = load_data()
    train_df, test_df = split_and_prepare_data(data)
    model = train_model(train_df, test_df, option)
