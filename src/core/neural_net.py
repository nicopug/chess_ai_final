from tensorflow import keras
from tensorflow.keras import layers, regularizers


def build_model():
    """Costruisce il modello con architettura ottimizzata"""
    # Input Layer
    board_input = keras.Input(shape=(8, 8, 12), name='board_input')

    # Feature Extraction (ridotta a 64 filtri)
    x = layers.Conv2D(64, (3, 3), padding='same',
                      activation='relu',
                      kernel_regularizer=regularizers.l2(0.0001))(board_input)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    # Solo 2 blocchi residui
    for _ in range(2):
        residual = x
        x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([x, residual])
        x = layers.Dropout(0.3)(x)

    # Policy Head con Attention
    policy_att = layers.MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
    policy_head = layers.Conv2D(128, (1, 1), activation='relu')(policy_att)
    policy_head = layers.Flatten()(policy_head)
    policy_output = layers.Dense(1968, activation='softmax', name='policy')(policy_head)

    # Value Head semplificata
    value_head = layers.GlobalAveragePooling2D()(x)
    value_head = layers.Dense(64, activation='swish')(value_head)
    value_head = layers.Dropout(0.3)(value_head)
    value_output = layers.Dense(1, activation='tanh', name='value')(value_head)

    # Model Compilation
    model = keras.Model(inputs=board_input, outputs=[policy_output, value_output])

    optimizer = keras.optimizers.Adam(
        learning_rate=0.001,  # Aumentato
        clipnorm=1.0
    )

    model.compile(
        optimizer=optimizer,
        loss={
            'policy': 'categorical_crossentropy',
            'value': 'mse'
        },
        loss_weights={
            'policy': 0.8,
            'value': 0.2
        },
        metrics={
            'policy': 'accuracy',
            'value': 'mae'
        }
    )
    print(model.summary())
    return model

build_model()