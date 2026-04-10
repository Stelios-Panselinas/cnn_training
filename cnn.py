import pandas as pd
import numpy as np
import os
import sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, GlobalAveragePooling1D, Input, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report

SEQ_LEN = 32
TRAIN_RATIO = 0.80
NUM_CLASSES = 3
FILTER1 = 8
FILTER2 = 16
KERNEL_SIZE = 3
POOL_SIZE = 2
DENSE = 4
BATCH = 4
EPOCHS = 10

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

        Dense(3, activation='softmax')
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

def split_and_prepare_data(data, option):
    subjects = data['subject'].unique()
    if(option == '0' or option == '1'):
        train_dfs = []
        test_dfs = []

        for sub in subjects:
            sub_data = data[data['subject'] == sub].reset_index(drop=True)
            sub_y_data = sub_data[label_col].values
            sub_x_data = sub_data[feature_cols].values
            sub_x_data, sub_y_data = create_windows(sub_x_data, sub_y_data, seq_len=SEQ_LEN)

            indices = np.random.permutation(sub_x_data.shape[0])
            sub_x_data = sub_x_data[indices]
            sub_y_data = sub_y_data[indices]

            slipt = int(len(sub_x_data) * TRAIN_RATIO)
            train_x_w = sub_x_data[:slipt]
            train_y_w = sub_y_data[:slipt]
            test_x_w = sub_x_data[slipt:]
            test_y_w = sub_y_data[slipt:]

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
        
        print(f'y test:{test_df['y'].unique()} y train:{train_df['y'].unique()}')
        print(f'distribution of y test: {test_df['y'].value_counts()} distribution of y train: {train_df['y'].value_counts()}')

        train_df.to_csv('train_df.csv')
        test_df.to_csv('test_df.csv')
        return train_df, test_df
    
    elif(option == '2'):
        test_sub = []
        test_sub.append(subjects[0])
        test_sub.append(subjects[1])
        test_sub.append(subjects[2])
        test_sub.append(subjects[4])
        test_sub.append(subjects[3])
        test_sub_set = set(test_sub)
        train_X = []
        train_Y = []
        test_X = []
        test_Y = []
        train_sub = list(filter(lambda x: x not in test_sub_set, subjects))
        for sub in train_sub:
            sub_data = data[data['subject'] == sub]
            train_x = sub_data[feature_cols].values
            train_y = sub_data[label_col].values
            train_x_w, train_y_w = create_windows(train_x, train_y, seq_len=SEQ_LEN)
            # train_x_w, test_x_w, scaler = normalize_data(train_x_w, test_x_w)
            train_X.append(train_x_w)
            train_Y.append(train_y_w)
        for sub in test_sub:
            sub_data = data[data['subject'] == sub]
            test_x = sub_data[feature_cols].values
            test_y = sub_data[label_col].values
            test_x_w, test_y_w = create_windows(test_x, test_y, seq_len=SEQ_LEN)
            # train_x_w, test_x_w, scaler = normalize_data(train_x_w, test_x_w)
            test_X.append(test_x_w)
            test_Y.append(test_y_w)
        train_X = np.concatenate(train_X)
        train_Y = np.concatenate(train_Y)
        test_X = np.concatenate(test_X)
        test_Y = np.concatenate(test_Y)
        train_X, test_X, scaler = normalize_data(train_X, test_X)
        train_df = pd.DataFrame({
                'subject': 'SUB',
                'X': list(train_X),   # store each window as array
                'y': train_Y
            })

        test_df = pd.DataFrame({
            'subject': 'SUB',
            'X': list(test_X),
            'y': test_Y
        })
        return train_df, test_df

def create_tflite_model_fp_io(model, train_X, output_path='model_int8_fp32_io.tflite'):
    def representative_data_gen():
        num_samples = min(100, len(train_X))
        for i in range(num_samples):
            sample = train_X[i:i+1].astype(np.float32)
            yield [sample]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimazations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen

    tflite_model_quant = converter.convert()

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
            epochs=EPOCHS,
            batch_size=BATCH,
            shuffle=True,
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
        epochs=EPOCHS,
        batch_size=BATCH,
        verbose=1
        )
        y_pred_probs = model.predict(test_X)
        y_pred = np.argmax(y_pred_probs, axis=1)
        cm = confusion_matrix(test_Y, y_pred)
        print("Confusion Matrix:\n", cm)
        print(classification_report(test_Y, y_pred, digits=4))
    elif(option == '2'):
        train_X = np.stack(train_df['X'].values)
        train_Y = train_df['y'].values

        test_X = np.stack(test_df['X'].values)
        test_Y = test_df['y'].values

        model = build_cnn(seq_len=SEQ_LEN, num_features=8, num_classes=NUM_CLASSES)
        history = model.fit(
        train_X, train_Y,
        validation_data=(test_X, test_Y),
        epochs=EPOCHS,
        batch_size=BATCH,
        verbose=1
        )
    return model

def export_tflite_model(model, train_X, output_path="model_int8.tflite"):
    def representative_data_gen():
        num_samples = min(100, len(train_X))
        for i in range(num_samples):
            sample = train_X[i:i+1].astype(np.float32)
            yield [sample]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    with open(output_path, "wb") as f:
        f.write(tflite_model)

    print(f"Saved fully quantized TFLite model: {output_path}")

    interpreter = tf.lite.Interpreter(model_path=output_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_scale, input_zero_point = input_details[0]['quantization']
    output_scale, output_zero_point = output_details[0]['quantization']

    print("Input scale:", input_scale)
    print("Input zero point:", input_zero_point)
    print("Output scale:", output_scale)
    print("Output zero point:", output_zero_point)

    return output_path

def export_wesad_eval_header(test_x, test_y, tflite_model_path, output_file="wesad_eval_data.h"):
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    input_scale, input_zero_point = input_details[0]['quantization']

    print("TFLite input scale:", input_scale)
    print("TFLite input zero point:", input_zero_point)

    test_x_q = np.round(test_x / input_scale + input_zero_point)
    test_x_q = np.clip(test_x_q, -128, 127).astype(np.int8)

    num_samples = test_x_q.shape[0]
    seq_len = test_x_q.shape[1]
    num_features = test_x_q.shape[2]
    sample_len = seq_len * num_features

    test_x_flat = test_x_q.reshape(num_samples, sample_len)

    with open(output_file, "w") as f:
        f.write("#ifndef WESAD_EVAL_DATA_H_\n")
        f.write("#define WESAD_EVAL_DATA_H_\n\n")
        f.write("#include <stdint.h>\n\n")

        f.write(f"const int wesad__num_samples = {num_samples};\n")
        f.write(f"const int g_wesad_eval_sample_len = {sample_len};\n")
        f.write(f"static const int wesad__values_per_sample = {sample_len};\n\n")

        f.write("const int8_t g_wesad_eval_x[] = {\n")
        for s in range(num_samples):
            f.write("  ")
            for i, val in enumerate(test_x_flat[s]):
                f.write(f"{int(val)}")
                if not (s == num_samples - 1 and i == sample_len - 1):
                    f.write(", ")
            f.write("\n")
        f.write("};\n\n")

        f.write("const uint8_t g_wesad_eval_y[] = {\n  ")
        for i, label in enumerate(test_y):
            f.write(f"{int(label)}")
            if i != len(test_y) - 1:
                f.write(", ")
        f.write("\n};\n\n")

        f.write("#endif\n")

    print(f"Saved eval header: {output_file}")

def test_quantized_tflite_model(tflite_model_path, test_X, test_Y, num_classes=3):
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_scale, input_zero_point = input_details[0]["quantization"]
    output_scale, output_zero_point = output_details[0]["quantization"]

    print("=== TFLite Quantized Model Test ===")
    print("Input dtype:", input_details[0]["dtype"])
    print("Input shape:", input_details[0]["shape"])
    print("Input scale:", input_scale)
    print("Input zero point:", input_zero_point)
    print("Output dtype:", output_details[0]["dtype"])
    print("Output shape:", output_details[0]["shape"])
    print("Output scale:", output_scale)
    print("Output zero point:", output_zero_point)

    preds = []

    for i in range(len(test_X)):
        sample = test_X[i:i+1].astype(np.float32)

        # quantize input exactly as TFLite expects
        sample_q = np.round(sample / input_scale + input_zero_point)
        sample_q = np.clip(sample_q, -128, 127).astype(np.int8)

        interpreter.set_tensor(input_details[0]["index"], sample_q)
        interpreter.invoke()

        output_data = interpreter.get_tensor(output_details[0]["index"])

        if output_details[0]["dtype"] == np.int8:
            pred = int(np.argmax(output_data[0]))
        else:
            pred = int(np.argmax(output_data[0]))

        preds.append(pred)

        if i < 5:
            print(f"\nSample {i}")
            print("True label:", int(test_Y[i]))
            print("Pred label:", pred)
            print("Quantized input first 16 vals:", sample_q.reshape(-1)[:16])
            print("Raw output:", output_data[0])

    preds = np.array(preds, dtype=np.int32)
    test_Y = np.array(test_Y, dtype=np.int32)

    acc = np.mean(preds == test_Y)
    print("\n=== Results ===")
    print(f"Accuracy: {acc * 100:.2f}%")

    pred_counts = np.bincount(preds, minlength=num_classes)
    label_counts = np.bincount(test_Y, minlength=num_classes)

    print("Label counts:", label_counts)
    print("Prediction counts:", pred_counts)

    print("\nConfusion Matrix:")
    print(confusion_matrix(test_Y, preds))

    print("\nClassification Report:")
    print(classification_report(test_Y, preds, digits=4))

    return preds

import re

def load_wesad_eval_header_int8(header_path, seq_len=32, num_features=8):
    with open(header_path, "r") as f:
        text = f.read()

    # metadata
    num_samples_match = re.search(r'wesad__num_samples\s*=\s*(\d+)', text)
    values_per_sample_match = re.search(r'wesad__values_per_sample\s*=\s*(\d+)', text)

    if not num_samples_match:
        raise ValueError("Could not find wesad__num_samples in header")
    if not values_per_sample_match:
        raise ValueError("Could not find wesad__values_per_sample in header")

    num_samples = int(num_samples_match.group(1))
    values_per_sample = int(values_per_sample_match.group(1))

    # g_wesad_eval_x
    x_match = re.search(
        r'const\s+int8_t\s+g_wesad_eval_x\[\]\s*=\s*\{(.*?)\};',
        text,
        re.S
    )
    if not x_match:
        raise ValueError("Could not find g_wesad_eval_x[] in header")

    x_text = x_match.group(1)
    x_vals = [int(v) for v in re.findall(r'-?\d+', x_text)]
    x_vals = np.array(x_vals, dtype=np.int8)

    expected_x_len = num_samples * values_per_sample
    if len(x_vals) != expected_x_len:
        raise ValueError(
            f"g_wesad_eval_x length mismatch: got {len(x_vals)}, expected {expected_x_len}"
        )

    # g_wesad_eval_y
    y_match = re.search(
        r'const\s+uint8_t\s+g_wesad_eval_y\[\]\s*=\s*\{(.*?)\};',
        text,
        re.S
    )
    if not y_match:
        raise ValueError("Could not find g_wesad_eval_y[] in header")

    y_text = y_match.group(1)
    y_vals = [int(v) for v in re.findall(r'\d+', y_text)]
    y_vals = np.array(y_vals, dtype=np.uint8)

    if len(y_vals) != num_samples:
        raise ValueError(
            f"g_wesad_eval_y length mismatch: got {len(y_vals)}, expected {num_samples}"
        )

    if values_per_sample != seq_len * num_features:
        raise ValueError(
            f"values_per_sample={values_per_sample}, but seq_len*num_features={seq_len * num_features}"
        )

    x_vals = x_vals.reshape(num_samples, seq_len, num_features)

    return x_vals, y_vals, num_samples, values_per_sample

def test_tflite_model_from_header(tflite_model_path, header_path, seq_len=32, num_features=8, num_classes=3):
    test_X_q, test_Y, num_samples, values_per_sample = load_wesad_eval_header_int8(
        header_path,
        seq_len=seq_len,
        num_features=num_features
    )

    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_scale, input_zero_point = input_details[0]["quantization"]
    output_scale, output_zero_point = output_details[0]["quantization"]

    print("=== TFLite Test From wesad_eval_data.h ===")
    print("Model:", tflite_model_path)
    print("Header:", header_path)
    print("Input dtype:", input_details[0]["dtype"])
    print("Input shape:", input_details[0]["shape"])
    print("Input scale:", input_scale)
    print("Input zero point:", input_zero_point)
    print("Output dtype:", output_details[0]["dtype"])
    print("Output shape:", output_details[0]["shape"])
    print("Output scale:", output_scale)
    print("Output zero point:", output_zero_point)
    print("Loaded samples:", num_samples)
    print("Values per sample:", values_per_sample)

    preds = []

    for i in range(num_samples):
        sample_q = test_X_q[i:i+1].astype(np.int8)

        interpreter.set_tensor(input_details[0]["index"], sample_q)
        interpreter.invoke()

        output_data = interpreter.get_tensor(output_details[0]["index"])
        pred = int(np.argmax(output_data[0]))
        preds.append(pred)

        if i < 5:
            print(f"\nSample {i}")
            print("True label:", int(test_Y[i]))
            print("Pred label:", pred)
            print("Header input first 16 vals:", sample_q.reshape(-1)[:16])
            print("Raw output:", output_data[0])

    preds = np.array(preds, dtype=np.int32)
    test_Y = np.array(test_Y, dtype=np.int32)

    acc = np.mean(preds == test_Y)

    print("\n=== Results ===")
    print(f"Accuracy: {acc * 100:.2f}%")

    label_counts = np.bincount(test_Y, minlength=num_classes)
    pred_counts = np.bincount(preds, minlength=num_classes)

    print("Label counts:", label_counts)
    print("Prediction counts:", pred_counts)

    print("\nConfusion Matrix:")
    print(confusion_matrix(test_Y, preds))

    print("\nClassification Report:")
    print(classification_report(test_Y, preds, digits=4))

    return preds, test_Y

if __name__ == "__main__":
    option = sys.argv[1]

    data, feature_cols, label_col = load_data()
    train_df, test_df = split_and_prepare_data(data, option)
    train_df = pd.read_csv('train_df.csv')
    test_df = pd.read_csv('test_df.csv')
    model = train_model(train_df, test_df, option)

    train_X = np.stack(train_df['X'].values).astype(np.float32)
    test_X  = np.stack(test_df['X'].values).astype(np.float32)
    test_Y  = test_df['y'].values.astype(np.uint8)

    tflite_path = export_tflite_model(model, train_X, output_path="model_int8.tflite")

    test_quantized_tflite_model(
        tflite_model_path="model_int8.tflite",
        test_X=test_X,
        test_Y=test_Y,
        num_classes=3
    )

    export_wesad_eval_header(
        test_X,
        test_Y,
        tflite_model_path=tflite_path,
        output_file="wesad_eval_data.h"
    )

    preds, labels = test_tflite_model_from_header(
    tflite_model_path="model_int8.tflite",
    header_path="wesad_eval_data.h",
    seq_len=32,
    num_features=8,
    num_classes=3
)
print(os.path.getsize("model_int8.tflite"))
