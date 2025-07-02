import os
import yaml
import numpy as np
import chess
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import (ModelCheckpoint,
                                        EarlyStopping,
                                        TensorBoard)
from src.core.config import DEBUG
import absl.logging
from datetime import datetime
from src.core.board_utils import board_to_array, move_to_index


# Configurazione
def load_config():
    """Carica il file di configurazione YAML"""
    config_path = 'C:/Users/domen/Desktop/chess_ai/configs/model_config.yaml'
    with open(config_path) as f:
        return yaml.safe_load(f)

# Callback per learning rate personalizzato
class SafeLRScheduler(keras.callbacks.Callback):
    def __init__(self, initial_lr, decay_steps, decay_rate):
        self.initial_lr = float(initial_lr)
        self.decay_steps = int(decay_steps)
        self.decay_rate = float(decay_rate)
        self.epoch_count = 0

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_count += 1
        if self.epoch_count % self.decay_steps == 0:
            new_lr = float(self.model.optimizer.learning_rate * self.decay_rate)
            self.model.optimizer.learning_rate.assign(new_lr)
            print(f"\n🚀 Adjusted LR to {new_lr:.4f}")

    def on_epoch_end(self, epoch, logs=None):
        logs['lr'] = float(self.model.optimizer.learning_rate.numpy())


# Generazione dati con augmentation
def generate_training_data(samples=15000, validation_split=0.15):
    """Genera dati bilanciati per training e validation"""
    # Valori pezzi standard
    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0  # Il re non conta nel materiale
    }

    boards, policies, values = [], [], []

    for _ in range(samples):
        try:
            # Inizializza scacchiera
            board = chess.Board()

            # Aggiungi mosse casuali (2-10)
            for _ in range(np.random.randint(2, 11)):
                if board.is_game_over():
                    break
                legal_moves = list(board.legal_moves)
                if legal_moves:
                    board.push(np.random.choice(legal_moves))

            # Calcola materiale (versione robusta)
            white_material = 0
            black_material = 0
            for square, piece in board.piece_map().items():
                if piece.color == chess.WHITE:
                    white_material += piece_values.get(piece.piece_type, 0)
                else:
                    black_material += piece_values.get(piece.piece_type, 0)
            material_diff = white_material - black_material

            # Calcolo valore posizionale
            center_control = sum(1 for sq in [chess.E4, chess.D4, chess.E5, chess.D5]
                                 if board.is_attacked_by(chess.WHITE, sq))
            value = np.tanh(center_control / 4 + material_diff / 12 + np.random.uniform(-0.1, 0.1))

            # Codifica board e policy
            board_array = board_to_array(board)
            policy = np.zeros(1968)

            if board.legal_moves:
                for move in board.legal_moves:
                    try:
                        idx = move_to_index(move) % 1968
                        # Assegna pesi differenziati
                        if move.to_square in [chess.E4, chess.D4, chess.E5, chess.D5]:
                            policy[idx] = np.random.uniform(0.7, 1.0)
                        elif board.is_capture(move):
                            policy[idx] = np.random.uniform(0.5, 0.8)
                        else:
                            policy[idx] = np.random.uniform(0.1, 0.3)
                    except Exception as e:
                        print(f"Errore codifica mossa {move}: {e}")
                        continue

                policy /= policy.sum()  # Normalizza

            boards.append(board_array)
            policies.append(policy)
            values.append(value)

        except Exception as e:
            print(f"Errore generazione dati: {e}")
            continue

    # Split train/validation
    split_idx = int(len(boards) * (1 - validation_split))
    return (\
        np.array(boards[:split_idx]), \
        np.array(policies[:split_idx]), \
        np.array(values[:split_idx]), \
        np.array(boards[split_idx:]), \
        np.array(policies[split_idx:]), \
        np.array(values[split_idx:])\
    )


# Costruzione modello
def build_model(config):
    """Costruisce il modello basato sulla configurazione"""
    # Input Layer
    board_input = keras.Input(shape=(8, 8, 12), name='board_input')

    # Feature Extraction
    x = layers.Conv2D(
        filters=config['model']['filters'],
        kernel_size=(3, 3),
        padding='same',
        activation='swish',
        kernel_regularizer=regularizers.l2(config['model']['l2_regularization'])
    )(board_input)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(config['model']['dropout_rate'])(x)

    # Blocchi Residui
    for _ in range(config['model']['residual_blocks']):
        residual = x
        x = layers.Conv2D(
            filters=config['model']['filters'],
            kernel_size=(3, 3),
            padding='same',
            activation='swish'
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(config['model']['dropout_rate'])(x)
        x = layers.Add()([x, residual])

    # Policy Head
    policy_head = layers.Conv2D(256, (1, 1), activation='swish')(x)
    policy_head = layers.Flatten()(policy_head)
    policy_head = layers.Dense(
        config['model']['policy_head_units'],
        activation='swish',
        kernel_regularizer=regularizers.l2(config['model']['l2_regularization'])
    )(policy_head)
    policy_output = layers.Dense(1968, activation='softmax', name='policy_head')(policy_head)

    # Value Head
    value_head = layers.GlobalAveragePooling2D()(x)
    value_head = layers.Dense(64, activation='swish')(value_head)
    value_head = layers.Dropout(0.3)(value_head)
    value_output = layers.Dense(1, activation='tanh', name='value_head')(value_head)

    # Model Compilation
    model = keras.Model(inputs=board_input, outputs=[policy_output, value_output])

    optimizer = keras.optimizers.Adam(
        learning_rate=float(config['training']['lr_schedule']['initial_learning_rate']),
        clipnorm=1.0
    )

    model.compile(
        optimizer=optimizer,
        loss={
            'policy_head': 'categorical_crossentropy',
            'value_head': 'mse'
        },
        loss_weights={
            'policy_head': 0.8,
            'value_head': 0.2
        },
        metrics={
            'policy_head': 'accuracy',
            'value_head': 'mae'
        }
    )

    return model


# Addestramento
def train_model():
    """Esegue il ciclo di addestramento completo"""
    config = load_config()
    model = build_model(config)

    train_boards, train_policies, train_values, val_boards, val_policies, val_values = generate_training_data(
        samples=config['data']['samples_per_epoch'],
        validation_split=config['training']['validation_split']
    )

    # Callbacks
    log_dir = os.path.join('../../logs', datetime.now().strftime('%Y%m%d-%H%M%S'))
    callbacks = [
        SafeLRScheduler(
            initial_lr=config['training']['lr_schedule']['initial_learning_rate'],
            decay_steps=config['training']['lr_schedule']['decay_steps'],
            decay_rate=config['training']['lr_schedule']['decay_rate']
        ),
        ModelCheckpoint(
            filepath='models/best_model.keras',
            monitor='val_policy_head_accuracy',
            save_best_only=True,
            mode='max',
        ),
        EarlyStopping(
            monitor='val_policy_head_accuracy',
            patience=10,
            restore_best_weights=True,
            mode='max',
            min_delta=0.003
        ),
        TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            profile_batch=(10, 20) if DEBUG else 0
        )
    ]

    # Disabilita log aggiuntivi
    absl.logging.set_verbosity(absl.logging.ERROR)

    # Addestramento
    history = model.fit(
        x=train_boards,
        y={'policy_head': train_policies, 'value_head': train_values},
        validation_data=(val_boards, {'policy_head': val_policies, 'value_head': val_values}),
        batch_size=config['training']['batch_size'],
        epochs=config['training']['epochs'],
        callbacks=callbacks,
        verbose=1
    )

    model.save('models/final_model.keras')
    return history

if __name__ == "__main__":
    train_model()