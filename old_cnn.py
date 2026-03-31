import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from sys import argv
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models, regularizers

tf.config.optimizer.set_jit(False)

# -----------------------------
# Hyperparameters
# -----------------------------
BATCH_SIZE = 32
FILTER1 = 3
FILTER2 = 4
SEQ_LEN = 16
POOL_SIZE = 3
KERNEL_SIZE = 4
DENSE_UNITS = 4
EPOCHS = 10

WESAD_PATH = "../wesad_extracted.csv"
SUBJECTS = ['S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S13', 'S14', 'S16', 'S17']
DOWNSAMPLE_FACTOR = 4

OUTPUT_DIR = "./tflite_models"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def reduce_wesad_classes(data, binary_classification=False, three_class_classification=False):
    if not binary_classification and not three_class_classification:
        return data

    selected_classes = sorted(data['label'].unique())[:3]
    data = data[data['label'].isin(selected_classes)]

    if binary_classification:
        data = data.replace({'label': 2}, {'label': 0})

    return data


def create_windows(signals, labels, seq_len):
    X, y = [], []

    for i in range(0, len(signals) - seq_len, seq_len):
        window_data = signals[i:i + seq_len]
        window_label_arr = labels[i:i + seq_len]
        window_label = np.bincount(window_label_arr).argmax()

        X.append(window_data)
        y.append(window_label)

    return np.array(X), np.array(y)


def load_data(select):
    raw_data = pd.read_csv(WESAD_PATH)
    downsampled_data = raw_data[::DOWNSAMPLE_FACTOR].reset_index(drop=True)

    downsampled_data = reduce_wesad_classes(
        downsampled_data,
        binary_classification=False,
        three_class_classification=True
    )

    labels = downsampled_data['label']
    subjects = downsampled_data['subject']

    if select == 'wrist':
        data = downsampled_data.drop(
            columns=[column for column in downsampled_data.columns if 'wrist' not in column]
        )
        data = pd.concat([data, labels, subjects], axis=1)
    else:
        data = downsampled_data.drop(
            columns=[column for column in downsampled_data.columns if 'wrist' in column]
        )
        if 'Unnamed: 0' in data.columns:
            data = data.drop(columns=['Unnamed: 0'])

    return data


def split_and_normalize_data(data, seq_len):
    scaler = StandardScaler()

    x = data.drop(columns=['label', 'subject']).values
    y = data['label'].values

    x_w, y_w = create_windows(x, y, seq_len=seq_len)

    train_x, test_x, train_y, test_y = train_test_split(
        x_w, y_w, test_size=0.2, random_state=42, shuffle=False
    )

    train_shape = train_x.shape
    test_shape = test_x.shape

    _, _, F = train_shape

    train_2d = train_x.reshape(-1, F)
    test_2d = test_x.reshape(-1, F)

    train_2d = scaler.fit_transform(train_2d)
    test_2d = scaler.transform(test_2d)

    train_x = train_2d.reshape(train_shape).astype(np.float32)
    test_x = test_2d.reshape(test_shape).astype(np.float32)

    return train_x, train_y, test_x, test_y, scaler


def build_cnn_model(input_shape, num_classes, f1, f2, kernel_size, pool_size, fc):
    model = models.Sequential([
        layers.Input(shape=input_shape),

        layers.Conv1D(
            filters=f1,
            kernel_size=kernel_size,
            activation='relu',
            kernel_regularizer=regularizers.l2(1e-4),
            padding='same'
        ),
        layers.MaxPooling1D(pool_size=pool_size),
        layers.Dropout(0.5),

        layers.Conv1D(
            filters=f2,
            kernel_size=kernel_size,
            activation='relu',
            kernel_regularizer=regularizers.l2(1e-4),
            padding='same'
        ),
        layers.MaxPooling1D(pool_size=pool_size),
        layers.Dropout(0.5),

        layers.GlobalAveragePooling1D(),

        layers.Dense(fc, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def make_representative_data_gen(train_x):
    def representative_data_gen():
        num_samples = min(100, len(train_x))
        for i in range(num_samples):
            sample = train_x[i:i+1].astype(np.float32)
            yield [sample]
    return representative_data_gen


def convert_to_full_int8_tflite(model, train_x, save_path):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Enable optimizations
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Representative dataset for full integer quantization
    converter.representative_dataset = make_representative_data_gen(train_x)

    # Force full integer quantization
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

    # Make model input/output int8 too
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    with open(save_path, "wb") as f:
        f.write(tflite_model)

    return tflite_model


def inspect_tflite_model(tflite_path):
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"\nTFLite model: {tflite_path}")
    print("Input details:", input_details)
    print("Output details:", output_details)


# -----------------------------
# Main
# -----------------------------
if len(argv) < 2:
    raise ValueError("Usage: python script.py wrist|chest")

select_mode = argv[1]
data = load_data(select=select_mode)

accuracies = []
tflite_paths = []

for sub in SUBJECTS:
    print(f"\n===== Training subject {sub} =====")

    sub_data = data[data['subject'] == sub].copy()
    train_x, train_y, test_x, test_y, scaler = split_and_normalize_data(sub_data, seq_len=SEQ_LEN)

    input_shape = (train_x.shape[1], train_x.shape[2])

    model = build_cnn_model(
        input_shape=input_shape,
        num_classes=3,
        f1=FILTER1,
        f2=FILTER2,
        kernel_size=KERNEL_SIZE,
        pool_size=POOL_SIZE,
        fc=DENSE_UNITS
    )

    history = model.fit(
        train_x,
        train_y,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(test_x, test_y),
        verbose=1
    )

    best_val_acc = float(np.max(history.history["val_accuracy"]))
    accuracies.append(best_val_acc)

    # Save float model if you want
    keras_path = os.path.join(OUTPUT_DIR, f"cnn_{select_mode}_{sub}.keras")
    model.save(keras_path)

    # Convert to full int8 TFLite
    tflite_path = os.path.join(OUTPUT_DIR, f"cnn_{select_mode}_{sub}_int8.tflite")
    convert_to_full_int8_tflite(model, train_x, tflite_path)
    tflite_paths.append(tflite_path)

    inspect_tflite_model(tflite_path)

print("\n===== Summary =====")
print("Per-subject best validation accuracies:", accuracies)
print("Average validation accuracy:", np.mean(accuracies))
print("Saved TFLite models:")
for p in tflite_paths:
    print(p)