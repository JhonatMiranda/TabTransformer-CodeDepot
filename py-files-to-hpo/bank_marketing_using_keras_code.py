import os

# Define a variável de ambiente CUDA_VISIBLE_DEVICES para usar apenas a CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import keras
from keras import layers

import math
import numpy as np
import pandas as pd
from tensorflow import data as tf_data
import matplotlib.pyplot as plt
from functools import partial
import tensorflow
import optuna

# Colunas do conjunto de dados Bank Marketing
CSV_HEADER = [
    "age", "job", "marital", "education", "default", "balance", "housing", "loan", "contact",
    "day", "month", "duration", "campaign", "pdays", "previous", "poutcome", "y"
]

# Carrega o conjunto de dados do CSV
bank_data = df = pd.read_csv("bank.csv", sep=';', header=None, names=CSV_HEADER, skiprows = 1)

# Exibe as primeiras linhas do DataFrame
bank_data.head()

from sklearn.model_selection import train_test_split

# Separar features (X) e rótulos (y)
X = bank_data[CSV_HEADER[:-1]]  # Todas as colunas, exceto a última (rótulo 'y')
y = bank_data['y']  # A última coluna (rótulo 'y')

# Dividir o conjunto de dados em treino e teste (80% para treino, 20% para teste)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Criar DataFrames para treino e teste
df_train = pd.concat([X_train, y_train], axis=1)
df_test = pd.concat([X_test, y_test], axis=1)

train_data_file = "train_data.csv"
test_data_file = "test_data.csv"

df_train.to_csv(train_data_file, index=False, header=False)
df_test.to_csv(test_data_file, index=False, header=False)

# A list of the numerical feature names.
NUMERIC_FEATURE_NAMES = [
    "age",
    "balance",
    "day",
    "duration",
    "campaign",
    "pdays",
    "previous",
]
# A dictionary of the categorical features and their vocabulary.
CATEGORICAL_FEATURES_WITH_VOCABULARY = {
    "job": sorted(list(df_train["job"].unique())),
    "marital": sorted(list(df_train["marital"].unique())),
    "education": sorted(list(df_train["education"].unique())),
    "default": sorted(list(df_train["default"].unique())),
    "housing": sorted(list(df_train["housing"].unique())),  # Assuming housing is binary
    "loan": sorted(list(df_train["loan"].unique())),  # Assuming loan is binary
    "contact": sorted(list(df_train["contact"].unique())),  # Assuming contact is binary
    "month": sorted(list(df_train["month"].unique())),
    "poutcome": sorted(list(df_train["poutcome"].unique())),
}
# A list of the categorical feature names.
CATEGORICAL_FEATURE_NAMES = list(CATEGORICAL_FEATURES_WITH_VOCABULARY.keys())
# A list of all the input features.
FEATURE_NAMES = NUMERIC_FEATURE_NAMES + CATEGORICAL_FEATURE_NAMES
# A list of column default values for each feature.
COLUMN_DEFAULTS = [
    [0.0] if feature_name in NUMERIC_FEATURE_NAMES else ["NA"]
    for feature_name in CSV_HEADER
]
# The name of the target feature.
TARGET_FEATURE_NAME = "y"
# A list of the labels of the target features.
TARGET_LABELS = ["no", "yes"]

"""#Hyperparameter Optimization"""

import numpy as np
from sklearn.metrics import roc_auc_score
from keras.callbacks import Callback


target_label_lookup = layers.StringLookup(
    vocabulary=TARGET_LABELS, mask_token=None, num_oov_indices=0
)


def prepare_example(features, target):
    target_index = target_label_lookup(target)
    return features, target_index,


lookup_dict = {}

for feature_name in CATEGORICAL_FEATURE_NAMES:
    vocabulary = CATEGORICAL_FEATURES_WITH_VOCABULARY[feature_name]
    # Create a lookup to convert a string values to an integer indices.
    # Since we are not using a mask token, nor expecting any out of vocabulary
    # (oov) token, we set mask_token to None and num_oov_indices to 0.
    lookup = layers.StringLookup(
        vocabulary=vocabulary, mask_token=None, num_oov_indices=0
    )
    lookup_dict[feature_name] = lookup


def encode_categorical(batch_x, batch_y):
    for feature_name in CATEGORICAL_FEATURE_NAMES:
        batch_x[feature_name] = lookup_dict[feature_name](batch_x[feature_name])

    return batch_x, batch_y


def get_dataset_from_csv(csv_file_path, batch_size=128, shuffle=False):
    dataset = (
        tf_data.experimental.make_csv_dataset(
            csv_file_path,
            batch_size=batch_size,
            column_names=CSV_HEADER,
            column_defaults=COLUMN_DEFAULTS,
            label_name=TARGET_FEATURE_NAME,
            num_epochs=1,
            header=False,
            na_value="?",
            shuffle=shuffle,
        )
        .map(prepare_example, num_parallel_calls=tf_data.AUTOTUNE, deterministic=False)
        .map(encode_categorical)
    )
    return dataset.cache()





def run_experiment(
    model,
    train_data_file,
    test_data_file,
    num_epochs,
    learning_rate,
    weight_decay,
    batch_size,
):
    optimizer = keras.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    )
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[keras.metrics.BinaryAccuracy(name="binary_accuracy")],
    )
    train_dataset = get_dataset_from_csv(train_data_file, batch_size, shuffle=True)
    validation_dataset = get_dataset_from_csv(test_data_file, batch_size)
    callbacks = [
    keras.callbacks.EarlyStopping(
        patience=15, monitor='val_loss')
    ]
    print("Start training the model...")
    history = model.fit(
        train_dataset,
        epochs=num_epochs,
        validation_data=validation_dataset,callbacks=callbacks
    )
    print("Model training finished")
    _, accuracy = model.evaluate(validation_dataset, verbose=0)
    print(f"Validation accuracy: {round(accuracy * 100, 2)}%")
    # Obter rótulos verdadeiros do conjunto de dados de validação
    y_val = []
    for batch in validation_dataset:
        y_val.append(batch[1].numpy())
    y_val = np.concatenate(y_val, axis=0)
    # Obter previsões do modelo no conjunto de dados de validação
    y_pred = model.predict(validation_dataset)
    # Calcular a AUC
    auc = roc_auc_score(y_val, y_pred)
    # print(f"AUC after training: {auc:.4f}")
    return history,auc
 

def create_model_inputs():
  inputs = {}
  for feature_name in FEATURE_NAMES:
      if feature_name in NUMERIC_FEATURE_NAMES:
          inputs[feature_name] = layers.Input(
              name=feature_name, shape=(), dtype="float32"
          )
      else:
          inputs[feature_name] = layers.Input(
              name=feature_name, shape=(), dtype="float32"
          )
  return inputs


def create_mlp(hidden_units, dropout_rate, activation, normalization_layer, name=None):
  mlp_layers = []
  for units in hidden_units:
      mlp_layers.append(normalization_layer()),
      mlp_layers.append(layers.Dense(units, activation=activation))
      mlp_layers.append(layers.Dropout(dropout_rate))
  return keras.Sequential(mlp_layers, name=name)


def encode_inputs(inputs, embedding_dims):
  encoded_categorical_feature_list = []
  numerical_feature_list = []
  for feature_name in inputs:
      if feature_name in CATEGORICAL_FEATURE_NAMES:
          vocabulary = CATEGORICAL_FEATURES_WITH_VOCABULARY[feature_name]
          # Create a lookup to convert a string values to an integer indices.
          # Since we are not using a mask token, nor expecting any out of vocabulary
          # (oov) token, we set mask_token to None and num_oov_indices to 0.
          # Convert the string input values into integer indices.
          # Create an embedding layer with the specified dimensions.
          embedding = layers.Embedding(
              input_dim=len(vocabulary), output_dim=embedding_dims
          )
          # Convert the index values to embedding representations.
          encoded_categorical_feature = embedding(inputs[feature_name])
          encoded_categorical_feature_list.append(encoded_categorical_feature)
      else:
          # Use the numerical features as-is.
          numerical_feature = keras.ops.expand_dims(inputs[feature_name], -1)
          numerical_feature_list.append(numerical_feature)
  return encoded_categorical_feature_list, numerical_feature_list

def create_tabtransformer_classifier(
  num_transformer_blocks,
  num_heads,
  embedding_dims,
  mlp_hidden_units_factors,
  dropout_rate,
  use_column_embedding=False,
):
  # Create model inputs.
  inputs = create_model_inputs()
  # encode features.
  encoded_categorical_feature_list, numerical_feature_list = encode_inputs(
      inputs, embedding_dims
  )
  # Stack categorical feature embeddings for the Tansformer.
  encoded_categorical_features = keras.ops.stack(encoded_categorical_feature_list, axis=1)
  # Concatenate numerical features.
  numerical_features = layers.concatenate(numerical_feature_list)
  # Add column embedding to categorical feature embeddings.
  if use_column_embedding:
      num_columns = encoded_categorical_features.shape[1]
      column_embedding = layers.Embedding(
          input_dim=num_columns, output_dim=embedding_dims
      )
      column_indices = keras.ops.arange(start=0, stop=num_columns, step=1)
      encoded_categorical_features = encoded_categorical_features + column_embedding(
          column_indices
      )
  # Create multiple layers of the Transformer block.
  for block_idx in range(num_transformer_blocks):
      # Create a multi-head attention layer.
      attention_output = layers.MultiHeadAttention(
          num_heads=num_heads,
          key_dim=embedding_dims,
          dropout=dropout_rate,
          name=f"multihead_attention_{block_idx}",
      )(encoded_categorical_features, encoded_categorical_features)
      # Skip connection 1.
      x = layers.Add(name=f"skip_connection1_{block_idx}")(
          [attention_output, encoded_categorical_features]
      )
      # Layer normalization 1.
      x = layers.LayerNormalization(name=f"layer_norm1_{block_idx}", epsilon=1e-6)(x)
      # Feedforward.
      feedforward_output = create_mlp(
          hidden_units=[embedding_dims],
          dropout_rate=dropout_rate,
          activation=keras.activations.gelu,
          normalization_layer=partial(
              layers.LayerNormalization, epsilon=1e-6
          ),  # using partial to provide keyword arguments before initialization
          name=f"feedforward_{block_idx}",
      )(x)
      # Skip connection 2.
      x = layers.Add(name=f"skip_connection2_{block_idx}")([feedforward_output, x])
      # Layer normalization 2.
      encoded_categorical_features = layers.LayerNormalization(
          name=f"layer_norm2_{block_idx}", epsilon=1e-6
      )(x)
  # Flatten the "contextualized" embeddings of the categorical features.
  categorical_features = layers.Flatten()(encoded_categorical_features)
  # Apply layer normalization to the numerical features.
  numerical_features = layers.LayerNormalization(epsilon=1e-6)(numerical_features)
  # Prepare the input for the final MLP block.
  features = layers.concatenate([categorical_features, numerical_features])
  # Compute MLP hidden_units.
  mlp_hidden_units = [
      factor * features.shape[-1] for factor in mlp_hidden_units_factors
  ]
  # Create final MLP.
  features = create_mlp(
      hidden_units=mlp_hidden_units,
      dropout_rate=dropout_rate,
      activation=keras.activations.selu,
      normalization_layer=layers.BatchNormalization,
      name="MLP",
  )(features)
  # Add a sigmoid as a binary classifer.
  outputs = layers.Dense(units=1, activation="sigmoid", name="sigmoid")(features)
  model = keras.Model(inputs=inputs, outputs=outputs)
  return model

def objective(trial):

  # Define hyperparameters to be optimized
  LEARNING_RATE_init = trial.suggest_float(
        "LEARNING_RATE", 1e-5, 1e-3
  )

  WEIGHT_DECAY_init = trial.suggest_float(
        "WEIGHT_DECAY", 1e-6, 1e-1
  )
  dropout_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
  dropout = trial.suggest_categorical('DROPOUT_RATE', dropout_values)


  LEARNING_RATE = LEARNING_RATE_init
  WEIGHT_DECAY = WEIGHT_DECAY_init
  DROPOUT_RATE = dropout
  BATCH_SIZE = 256
  NUM_EPOCHS = 300

  NUM_TRANSFORMER_BLOCKS = 6  # Number of transformer blocks.
  NUM_HEADS = 8  # Number of attention heads.
  EMBEDDING_DIMS = 32  # Embedding dimensions of the categorical features.
  MLP_HIDDEN_UNITS_FACTORS = [
      4,
      2,
  ]  # MLP hidden layer units, as factors of the number of inputs.
  NUM_MLP_BLOCKS = 2  # Number of MLP blocks in the baseline model.


  tabtransformer_model = create_tabtransformer_classifier(
      num_transformer_blocks=NUM_TRANSFORMER_BLOCKS,
      num_heads=NUM_HEADS,
      embedding_dims=EMBEDDING_DIMS,
      mlp_hidden_units_factors=MLP_HIDDEN_UNITS_FACTORS,
      dropout_rate=DROPOUT_RATE,
  )

  print("Total model weights:", tabtransformer_model.count_params())
  # keras.utils.plot_model(tabtransformer_model, show_shapes=True, rankdir="LR")

  history,auc = run_experiment(
    model=tabtransformer_model,
    train_data_file=train_data_file,
    test_data_file=test_data_file,
    num_epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    batch_size=BATCH_SIZE,
  )
  # Return the mean AUC score over all folds
  return auc


# Optimize hyperparameters using optuna
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)
# Get the best hyperparameters
best_params = study.best_params
print("Best Hyperparameters:", best_params)