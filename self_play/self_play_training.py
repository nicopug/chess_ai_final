import os
import sys
import time
import json
import random
import logging
import datetime
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

# Importa i moduli personalizzati
from self_play import SelfPlayGenerator, create_neural_network_model
from data_preparation import prepare_training_data
from incremental_trainer import train_network
from evaluator import evaluate_model

# Configurazione dei path
PROJECT_ROOT = Path(os.path.dirname(os.path.abspath(__file__))).parent
BASE_MODEL_PATH = PROJECT_ROOT / "models" / "base_model"
SELF_PLAY_MODELS_DIR = PROJECT_ROOT / "models" / "self_play"
GAME_DATA_DIR = PROJECT_ROOT / "data" / "games"
LOGS_DIR = PROJECT_ROOT / "logs"

# Creazione delle directory se non esistono
for dir_path in [BASE_MODEL_PATH, SELF_PLAY_MODELS_DIR, GAME_DATA_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Configurazione del logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / f"training_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('self_play_training')

# Configurazione del training
CONFIG = {
    "num_iterations": 20,  # Numero di iterazioni di self-play
    "num_games_per_iteration": 50,  # Partite per iterazione
    "mcts_simulations": 800,  # Simulazioni MCTS per mossa
    "temperature": 1.0,  # Temperatura per la selezione delle mosse
    "temperature_drop_move": 10,  # Mossa dopo cui abbassare la temperatura a 0
    "batch_size": 512,  # Dimensione batch per il training
    "epochs": 10,  # Epoche per ogni sessione di training
    "learning_rate": 0.001,  # Learning rate iniziale
    "lr_schedule": {  # Schedule per il learning rate
        "5": 0.0005,
        "10": 0.0001,
        "15": 0.00005
    },
    "evaluation": {
        "enabled": True,  # Abilita la valutazione del modello
        "num_games": 20,  # Numero di partite per la valutazione
        "win_threshold": 0.55  # Percentuale di vittorie necessaria per accettare il nuovo modello
    }
}


def plot_training_metrics(metrics_history, save_path=None):
    """
    Visualizza e salva i grafici delle metriche di training.

    Args:
        metrics_history: Dizionario con le metriche di training
        save_path: Path dove salvare i grafici
    """
    plt.figure(figsize=(15, 10))

    # Plot della loss
    plt.subplot(2, 2, 1)
    plt.plot(metrics_history['policy_loss'], label='Policy Loss')
    plt.plot(metrics_history['value_loss'], label='Value Loss')
    plt.plot(metrics_history['total_loss'], label='Total Loss')
    plt.title('Training Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot dell'accuratezza
    plt.subplot(2, 2, 2)
    plt.plot(metrics_history['policy_accuracy'], label='Policy Accuracy')
    plt.title('Policy Accuracy')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Plot delle valutazioni
    if 'win_rate' in metrics_history and len(metrics_history['win_rate']) > 0:
        plt.subplot(2, 2, 3)
        plt.plot(metrics_history['win_rate'], label='Win Rate vs Old Model')
        plt.axhline(y=CONFIG['evaluation']['win_threshold'], color='r', linestyle='--',
                    label=f'Threshold ({CONFIG["evaluation"]["win_threshold"]})')
        plt.title('Evaluation Win Rate')
        plt.xlabel('Iteration')
        plt.ylabel('Win Rate')
        plt.legend()
        plt.grid(True)

    # Plot del learning rate
    plt.subplot(2, 2, 4)
    plt.plot(metrics_history['learning_rate'], label='Learning Rate')
    plt.title('Learning Rate')
    plt.xlabel('Iteration')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        logger.info(f"Grafici delle metriche salvati in {save_path}")

    plt.show()


def save_iteration_results(iteration, model, metrics, games_data=None):
    """
    Salva i risultati di un'iterazione di training.

    Args:
        iteration: Numero dell'iterazione
        model: Modello da salvare
        metrics: Metriche di training da salvare
        games_data: Dati delle partite da salvare (opzionale)
    """
    # Salva il modello
    model_save_path = SELF_PLAY_MODELS_DIR / f"model_iteration_{iteration}"
    model.save(model_save_path)
    logger.info(f"Modello dell'iterazione {iteration} salvato in {model_save_path}")

    # Salva le metriche
    metrics_save_path = LOGS_DIR / f"metrics_iteration_{iteration}.json"
    with open(metrics_save_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Metriche dell'iterazione {iteration} salvate in {metrics_save_path}")

    # Salva i dati delle partite
    if games_data:
        games_save_path = GAME_DATA_DIR / f"games_iteration_{iteration}.npz"

        # Estrai e converti i dati per il salvataggio
        states = [game['states'] for game in games_data]
        policies = [game['policies'] for game in games_data]
        results = [game['result'] for game in games_data]

        np.savez_compressed(
            games_save_path,
            states=np.array(states, dtype=object),
            policies=np.array(policies, dtype=object),
            results=np.array(results)
        )
        logger.info(f"Dati delle partite dell'iterazione {iteration} salvati in {games_save_path}")


def main():
    """
    Funzione principale per il training con self-play.
    """
    # Inizializza la cronologia delle metriche
    metrics_history = {
        'policy_loss': [],
        'value_loss': [],
        'total_loss': [],
        'policy_accuracy': [],
        'learning_rate': [],
        'win_rate': []
    }

    # Carica o crea il modello base
    if os.path.exists(BASE_MODEL_PATH):
        model = tf.keras.models.load_model(BASE_MODEL_PATH)
        logger.info(f"Modello base caricato da {BASE_MODEL_PATH}")
    else:
        model = create_neural_network_model()
        model.save(BASE_MODEL_PATH)
        logger.info(f"Nuovo modello base creato e salvato in {BASE_MODEL_PATH}")

    # Ciclo principale di training
    for iteration in range(1, CONFIG["num_iterations"] + 1):
        start_time = time.time()
        logger.info(f"Iniziando iterazione {iteration}/{CONFIG['num_iterations']}")

        # Aggiorna il learning rate se necessario
        if str(iteration) in CONFIG["lr_schedule"]:
            new_lr = CONFIG["lr_schedule"][str(iteration)]
            tf.keras.backend.set_value(model.optimizer.learning_rate, new_lr)
            logger.info(f"Learning rate aggiornato a {new_lr}")

        current_lr = float(tf.keras.backend.get_value(model.optimizer.learning_rate))
        metrics_history['learning_rate'].append(current_lr)

        # 1. Generazione dei dati attraverso self-play
        logger.info("Generazione delle partite attraverso self-play...")
        self_play_generator = SelfPlayGenerator(
            model=model,
            num_games=CONFIG["num_games_per_iteration"],
            mcts_simulations=CONFIG["mcts_simulations"],
            temperature=CONFIG["temperature"],
            temperature_drop_move=CONFIG["temperature_drop_move"]
        )
        games_data = self_play_generator.generate_games()
        logger.info(f"Generate {len(games_data)} partite")

        # 2. Preparazione dei dati di training
        logger.info("Preparazione dei dati di training...")
        training_data = prepare_training_data(games_data)
        logger.info(f"Preparati {len(training_data['states'])} esempi di training")

        # 3. Training del modello
        logger.info("Training del modello...")
        training_history = train_network(
            model=model,
            training_data=training_data,
            batch_size=CONFIG["batch_size"],
            epochs=CONFIG["epochs"]
        )

        # Aggiorna la cronologia delle metriche
        metrics_history['policy_loss'].append(training_history['policy_loss'][-1])
        metrics_history['value_loss'].append(training_history['value_loss'][-1])
        metrics_history['total_loss'].append(training_history['loss'][-1])
        metrics_history['policy_accuracy'].append(training_history['policy_accuracy'][-1])

        # 4. Valutazione del modello (opzionale)
        if CONFIG["evaluation"]["enabled"]:
            logger.info("Valutazione del modello...")
            # Carica il modello precedente per il confronto
            if iteration > 1:
                prev_model_path = SELF_PLAY_MODELS_DIR / f"model_iteration_{iteration - 1}"
                if os.path.exists(prev_model_path):
                    prev_model = tf.keras.models.load_model(prev_model_path)

                    win_rate = evaluate_model(
                        new_model=model,
                        old_model=prev_model,
                        num_games=CONFIG["evaluation"]["num_games"]
                    )

                    metrics_history['win_rate'].append(win_rate)
                    logger.info(f"Tasso di vittoria contro il modello precedente: {win_rate:.2%}")

                    # Decide se accettare il nuovo modello
                    if win_rate < CONFIG["evaluation"]["win_threshold"]:
                        logger.warning(
                            f"Il nuovo modello non ha superato la soglia di vittoria ({win_rate:.2%} < {CONFIG['evaluation']['win_threshold']:.2%})")
                        # Opzione: ripristina il modello precedente
                        # model = prev_model
                        # logger.info("Ripristinato il modello precedente")
            else:
                # Prima iterazione, nessun modello precedente da confrontare
                metrics_history['win_rate'].append(0.5)  # Valore neutro per il plotting

        # 5. Salvataggio dei risultati
        iteration_metrics = {
            'iteration': iteration,
            'policy_loss': float(metrics_history['policy_loss'][-1]),
            'value_loss': float(metrics_history['value_loss'][-1]),
            'total_loss': float(metrics_history['total_loss'][-1]),
            'policy_accuracy': float(metrics_history['policy_accuracy'][-1]),
            'learning_rate': float(metrics_history['learning_rate'][-1]),
            'win_rate': float(metrics_history['win_rate'][-1]) if metrics_history['win_rate'] else None,
            'elapsed_time': time.time() - start_time
        }

        save_iteration_results(iteration, model, iteration_metrics, games_data)

        # Visualizza il progresso
        logger.info(f"Iterazione {iteration} completata in {iteration_metrics['elapsed_time']:.2f} secondi")
        logger.info(f"Metriche: Loss={iteration_metrics['total_loss']:.4f}, "
                    f"Policy Accuracy={iteration_metrics['policy_accuracy']:.4f}")

        # Visualizza i grafici ogni 5 iterazioni o all'ultima iterazione
        if iteration % 5 == 0 or iteration == CONFIG["num_iterations"]:
            plot_save_path = LOGS_DIR / f"training_metrics_iteration_{iteration}.png"
            plot_training_metrics(metrics_history, save_path=plot_save_path)

    # Salva il modello finale
    final_model_path = SELF_PLAY_MODELS_DIR / "final_model"
    model.save(final_model_path)
    logger.info(f"Training completato. Modello finale salvato in {final_model_path}")

    # Visualizza un riepilogo finale
    logger.info("\n== Riepilogo del training ==")
    logger.info(f"Iterazioni completate: {CONFIG['num_iterations']}")
    logger.info(f"Partite totali generate: {CONFIG['num_iterations'] * CONFIG['num_games_per_iteration']}")
    logger.info(
        f"Loss iniziale vs finale: {metrics_history['total_loss'][0]:.4f} -> {metrics_history['total_loss'][-1]:.4f}")
    logger.info(
        f"Accuratezza policy iniziale vs finale: {metrics_history['policy_accuracy'][0]:.4f} -> {metrics_history['policy_accuracy'][-1]:.4f}")
    if metrics_history['win_rate']:
        logger.info(f"Tasso di vittoria finale: {metrics_history['win_rate'][-1]:.2%}")

    # Visualizza i grafici finali
    final_plot_path = LOGS_DIR / "final_training_metrics.png"
    plot_training_metrics(metrics_history, save_path=final_plot_path)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(f"Errore durante l'esecuzione del training: {e}")
        raise
