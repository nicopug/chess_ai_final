import tensorflow as tf
import numpy as np
import time
import logging
from tqdm import tqdm

# Configurazione del logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('incremental_trainer')


def train_network(model, training_data, batch_size=512, epochs=10, validation_split=0.1):
    """
    Allena la rete neurale con i dati di training forniti.

    Args:
        model: Modello TensorFlow/Keras da allenare
        training_data: Dizionario con stati, policy_targets e value_targets
        batch_size: Dimensione del batch per il training
        epochs: Numero di epoche da eseguire
        validation_split: Frazione dei dati da usare per la validazione

    Returns:
        Dizionario con la storia del training
    """
    states = training_data['states']
    policy_targets = training_data['policy_targets']
    value_targets = training_data['value_targets']

    logger.info(f"Training del modello con {len(states)} esempi, batch_size={batch_size}, epochs={epochs}")

    # Definisce le callback per il training
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            verbose=1,
            restore_best_weights=True
        )
    ]

    # Allena il modello
    start_time = time.time()
    history = model.fit(
        x=states,
        y={
            'policy': policy_targets,
            'value': value_targets
        },
        batch_size=batch_size,
        epochs=epochs,
        validation_split=validation_split,
        callbacks=callbacks,
        verbose=1
    )
    training_time = time.time() - start_time

    logger.info(f"Training completato in {training_time:.2f} secondi")

    # Estrai le metriche dalla storia del training
    metrics = {
        'loss': history.history['loss'],
        'policy_loss': history.history['policy_loss'],
        'value_loss': history.history['value_loss'],
        'policy_accuracy': history.history['policy_accuracy'],
        'val_loss': history.history['val_loss'],
        'val_policy_loss': history.history['val_policy_loss'],
        'val_value_loss': history.history['val_value_loss'],
        'val_policy_accuracy': history.history['val_policy_accuracy'],
        'epochs': len(history.history['loss']),
        'training_time': training_time
    }

    # Mostra il riepilogo delle metriche finali
    logger.info(f"Metriche finali:")
    logger.info(f"  Loss: {metrics['loss'][-1]:.4f} (val: {metrics['val_loss'][-1]:.4f})")
    logger.info(f"  Policy Loss: {metrics['policy_loss'][-1]:.4f} (val: {metrics['val_policy_loss'][-1]:.4f})")
    logger.info(f"  Value Loss: {metrics['value_loss'][-1]:.4f} (val: {metrics['val_value_loss'][-1]:.4f})")
    logger.info(
        f"  Policy Accuracy: {metrics['policy_accuracy'][-1]:.4f} (val: {metrics['val_policy_accuracy'][-1]:.4f})")

    return metrics


def train_on_memory_buffer(model, memory_buffer, batch_size=512, min_samples=1000, epochs=1):
    """
    Allena il modello su un buffer di memoria (utile per l'apprendimento per rinforzo).

    Args:
        model: Modello TensorFlow/Keras da allenare
        memory_buffer: Lista di tuple (stato, policy, valore)
        batch_size: Dimensione del batch per il training
        min_samples: Numero minimo di campioni necessari per iniziare il training
        epochs: Numero di epoche da eseguire

    Returns:
        Dizionario con la storia del training o None se non ci sono abbastanza campioni
    """
    if len(memory_buffer) < min_samples:
        logger.info(f"Non ci sono abbastanza campioni nel buffer ({len(memory_buffer)} < {min_samples})")
        return None

    # Estrai i dati dal buffer
    states = []
    policy_targets = []
    value_targets = []

    for state, policy, value in memory_buffer:
        states.append(state)
        policy_targets.append(policy)
        value_targets.append(value)

    # Converti in array numpy
    states = np.array(states)
    policy_targets = np.array(policy_targets)
    value_targets = np.array(value_targets).reshape(-1, 1)

    # Prepara i dati di training
    training_data = {
        'states': states,
        'policy_targets': policy_targets,
        'value_targets': value_targets
    }

    # Allena il modello
    return train_network(model, training_data, batch_size, epochs)


def fine_tune_model(base_model, new_data, learning_rate=0.001, batch_size=256, epochs=5):
    """
    Fine-tune un modello esistente con nuovi dati.

    Args:
        base_model: Modello base da fine-tunare
        new_data: Nuovi dati di training
        learning_rate: Learning rate per il fine-tuning
        batch_size: Dimensione del batch
        epochs: Numero di epoche

    Returns:
        Modello fine-tunato e metriche di training
    """
    # Crea una copia del modello base
    fine_tuned_model = tf.keras.models.clone_model(base_model)
    fine_tuned_model.set_weights(base_model.get_weights())

    # Compila il modello con un learning rate più basso
    fine_tuned_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss={
            'policy': 'categorical_crossentropy',
            'value': 'mean_squared_error'
        },
        metrics={
            'policy': 'accuracy'
        }
    )

    # Fine-tune il modello
    metrics = train_network(
        model=fine_tuned_model,
        training_data=new_data,
        batch_size=batch_size,
        epochs=epochs
    )

    return fine_tuned_model, metrics


def create_learning_rate_scheduler(initial_lr=0.001, decay_steps=10000, decay_rate=0.96):
    """
    Crea uno scheduler per il learning rate che lo riduce gradualmente.

    Args:
        initial_lr: Learning rate iniziale
        decay_steps: Dopo quanti step ridurre il learning rate
        decay_rate: Fattore di riduzione

    Returns:
        Funzione di scheduling del learning rate
    """

    def lr_scheduler(epoch, lr):
        if epoch > 0 and epoch % decay_steps == 0:
            return lr * decay_rate
        return lr

    return lr_scheduler


if __name__ == "__main__":
    # Test per verificare il funzionamento
    from self_play import create_neural_network_model
    import numpy as np

    # Crea un modello di test
    model = create_neural_network_model()

    # Crea dati di test
    num_samples = 1000
    states = np.random.rand(num_samples, 8, 8, 12)
    policy_targets = np.random.rand(num_samples, 1968)
    policy_targets = policy_targets / policy_targets.sum(axis=1, keepdims=True)  # Normalizza
    value_targets = np.random.uniform(-1, 1, (num_samples, 1))

    training_data = {
        'states': states,
        'policy_targets': policy_targets,
        'value_targets': value_targets
    }

    # Testa il training
    metrics = train_network(model, training_data, batch_size=16, epochs=1)

    print(f"Test completato con successo. Loss finale: {metrics['loss'][-1]:.4f}")
