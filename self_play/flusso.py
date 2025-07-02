import os
import sys

# Aggiungi la root del progetto al path per risolvere i problemi di importazione
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import tensorflow as tf
import numpy as np
import logging
import matplotlib.pyplot as plt
from datetime import datetime
import time
import chess
import concurrent.futures

from self_play import create_neural_network_model, SelfPlayGenerator
from data_preparation import prepare_training_data, balance_training_data
from incremental_trainer import train_network, fine_tune_model
from evaluator import evaluate_model, evaluate_with_elo_system, analyze_positions

# Configurazione del logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('training_pipeline')

# Usa le stesse costanti definite in self_play_training.py
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
BASE_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "base_model")
SELF_PLAY_MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "self_play")
GAME_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "games")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")

# Crea le directory se non esistono
os.makedirs(SELF_PLAY_MODELS_DIR, exist_ok=True)
os.makedirs(GAME_DATA_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# Parametri di configurazione
CONFIG = {
    "num_iterations": 10,
    "games_per_iteration": 10, # Aumentato per un dataset più ricco
    "mcts_simulations": 800,
    "training_epochs": 10,
    "batch_size": 256,
    "learning_rate": 0.001,
    "dirichlet_alpha": 0.5, # Aumentato per maggiore esplorazione
    "dirichlet_weight": 0.25,
    "temperature": 1.0,
    "temperature_drop_move": 30,
    "evaluation_games": 40,
    "win_rate_threshold": 0.55,
    "c_puct": 4.0, # Aumentato per maggiore esplorazione
    "num_workers": max(1, os.cpu_count() - 1)  # Usa tutti i core tranne uno per non sovraccaricare il sistema
}


def plot_training_metrics(iteration, metrics):
    """
    Visualizza i grafici delle metriche di training
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.figure(figsize=(15, 10))

    # Plot per la loss totale
    plt.subplot(2, 2, 1)
    plt.plot(metrics['loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')

    # Plot per la loss della policy
    plt.subplot(2, 2, 2)
    plt.plot(metrics['policy_head_loss'])
    plt.title('Policy Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')

    # Plot per la loss del valore
    plt.subplot(2, 2, 3)
    plt.plot(metrics['value_head_loss'])
    plt.title('Value Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')

    # Plot per l'accuratezza della policy
    plt.subplot(2, 2, 4)
    plt.plot(metrics['policy_head_accuracy'])
    plt.title('Policy Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')

    plt.tight_layout()

    # Salva il grafico
    plot_path = os.path.join(LOGS_DIR, f"metrics_iter_{iteration}_{timestamp}.png")
    plt.savefig(plot_path)
    plt.close()

    logger.info(f"Metriche di training salvate in: {plot_path}")


def save_iteration_results(iteration, metrics, win_rate, elo_gain=None):
    """
    Salva i risultatidi un'iterazione in un file di testo
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(LOGS_DIR, f"iteration_{iteration}_results_{timestamp}.txt")

    with open(results_file, 'w') as f:
        f.write(f"=== Risultati Iterazione {iteration} ===\n\n")
        f.write(f"Data e ora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("Parametri di configurazione:\n")
        for key, value in CONFIG.items():
            f.write(f"  {key}: {value}\n")
        f.write("\n")

        f.write("Metriche di training finali:\n")
        f.write(f"  Loss totale: {metrics['loss'][-1]:.4f}\n")
        f.write(f"  Policy loss: {metrics['policy_head_loss'][-1]:.4f}\n")
        f.write(f"  Value loss: {metrics['value_head_loss'][-1]:.4f}\n")
        f.write(f"  Policy accuracy: {metrics['policy_head_accuracy'][-1]:.4f}\n\n")

        f.write("Risultati valutazione:\n")
        f.write(f"  Win rate vs modello precedente: {win_rate:.4f}\n")
        if elo_gain is not None:
            f.write(f"  Guadagno Elo stimato: {elo_gain:.1f}\n\n")

        f.write("Note:\n")
        if win_rate >= CONFIG["win_rate_threshold"]:
            f.write("  Il nuovo modello è stato accettato come modello di riferimento.\n")
        else:
            f.write("  Il nuovo modello NON ha superato la soglia richiesta e non è stato accettato.\n")

    logger.info(f"Risultati dell'iterazione salvati in: {results_file}")


def game_generation_worker(model_path, config, game_idx):
    """
    Worker function to generate a single game in a separate process.
    """
    logger.info(f"Avvio processo worker per la partita {game_idx + 1}...")
    try:
        # Carica il modello all'interno del processo worker per evitare problemi di serializzazione
        model = tf.keras.models.load_model(model_path)

        generator = SelfPlayGenerator(
            model=model,
            num_games=1,  # Genera una partita per processo
            mcts_simulations=config["mcts_simulations"],
            temperature=config["temperature"],
            temperature_drop_move=config["temperature_drop_move"],
            dirichlet_alpha=config["dirichlet_alpha"],
            dirichlet_weight=config["dirichlet_weight"],
            c_puct=config["c_puct"]
        )
        return generator.generate_games()
    except Exception as e:
        logger.error(f"Errore nel worker {game_idx}: {e}", exc_info=True)
        return []


def main():
    """
    Funzione principale che gestisce l'intero ciclo di training
    """
    start_time = time.time()
    logger.info("Avvio del ciclo di training AlphaZero")

    # Definisci alcune posizioni di test in formato FEN
    test_positions_fen = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",  # Posizione iniziale
        "r1bqkbnr/pp1ppppp/2n5/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",  # Difesa Siciliana
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",  # Mediogioco complesso
        "8/k7/3p4/p2P1p2/P2P1P2/8/8/K7 w - - 0 1"  # Finale di re e pedoni
    ]

    # Crea il modello iniziale o carica l'ultimo salvato
    latest_model_path = os.path.join(SELF_PLAY_MODELS_DIR, "latest_model.keras")

    if os.path.exists(latest_model_path):
        logger.info("Caricamento dell'ultimo modello salvato...")
        best_model = tf.keras.models.load_model(latest_model_path)
    else:
        logger.info("Creazione di un nuovo modello base...")
        best_model = create_neural_network_model()
        best_model.save(latest_model_path)
        logger.info(f"Nuovo modello salvato in: {latest_model_path}")

    # Ciclo principale di training
    for iteration in range(1, CONFIG["num_iterations"] + 1):
        iteration_start = time.time()
        logger.info(f"=== Iterazione {iteration}/{CONFIG['num_iterations']} ===")

        # 1. Genera partite con self-play in parallelo
        logger.info(f"Generazione di {CONFIG['games_per_iteration']} partite con self-play in parallelo...")
        games_data = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=CONFIG["num_workers"]) as executor:
            futures = [executor.submit(game_generation_worker, latest_model_path, CONFIG, i) for i in range(CONFIG["games_per_iteration"])]
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        games_data.extend(result)
                except Exception as e:
                    logger.error(f"Errore durante la raccolta dei risultati della partita: {e}", exc_info=True)

        if not games_data:
            logger.error("Nessun dato di partita generato. Interruzione del training.")
            break

        # Salva i dati delle partite
        games_path = os.path.join(GAME_DATA_DIR, f"games_iter_{iteration}.npz")
        np.savez_compressed(games_path, games_data=games_data)
        logger.info(f"Dati delle partite salvati in: {games_path}")

        # 2. Prepara i dati per il training
        logger.info("Preparazione dei dati di training...")
        training_data = prepare_training_data(games_data)
        balanced_data = balance_training_data(training_data)

        # 3. Crea un nuovo modello clonando il migliore
        new_model = tf.keras.models.clone_model(best_model)
        new_model.set_weights(best_model.get_weights())
        new_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=CONFIG["learning_rate"]),
            loss={
                'policy_head': 'categorical_crossentropy',
                'value_head': 'mean_squared_error'
            },
            metrics={
                'policy_head': 'accuracy',
                'value_head': 'mean_squared_error'
            }
        )

        # 4. Allena il nuovo modello
        logger.info("Training del nuovo modello...")
        metrics = train_network(
            model=new_model,
            training_data=balanced_data,
            batch_size=CONFIG["batch_size"],
            epochs=CONFIG["training_epochs"]
        )

        # Visualizza e salva le metriche di training
        plot_training_metrics(iteration, metrics)

        # 5. Salva il modello candidato
        candidate_model_path = os.path.join(SELF_PLAY_MODELS_DIR, f"model_iter_{iteration}.keras")
        new_model.save(candidate_model_path)
        logger.info(f"Modello candidato salvato in: {candidate_model_path}")

        # 6. Valuta il nuovo modello contro il migliore attuale
        logger.info("Valutazione del nuovo modello...")
        win_rate = evaluate_model(
            new_model=new_model,
            old_model=best_model,
            num_games=CONFIG["evaluation_games"],
            mcts_simulations=CONFIG["mcts_simulations"]
        )

        # Opzionale: valuta con il sistema Elo
        try:
            elo_gain = evaluate_with_elo_system(
                models={"new_model": new_model, "best_model": best_model},
                num_games_per_pair=CONFIG["evaluation_games"] // 2
            )
        except Exception as e:
            logger.warning(f"Errore durante la valutazione Elo: {str(e)}")
            elo_gain = None

        # 7. Analizza il modello su posizioni test (opzionale)
        try:
            logger.info("Analisi del modello su posizioni di test...")
            analyze_positions(new_model, positions=test_positions_fen)
        except Exception as e:
            logger.warning(f"Errore durante l'analisi delle posizioni: {str(e)}")

        # 8. Decidi se sostituire il miglior modello
        if win_rate >= CONFIG["win_rate_threshold"]:
            logger.info(f"Nuovo modello accettato con win_rate = {win_rate:.2f}")
            best_model = new_model
            best_model.save(latest_model_path)
            # Salva anche una copia numerata del miglior modello
            best_model_path = os.path.join(SELF_PLAY_MODELS_DIR, f"best_model_iter_{iteration}.keras")
            best_model.save(best_model_path)
            logger.info(f"Miglior modello aggiornato e salvato in: {latest_model_path} e {best_model_path}")
        else:
            logger.info(f"Nuovo modello rifiutato. Win rate: {win_rate:.2f}")

        # 9. Salva i risultati dell'iterazione
        save_iteration_results(iteration, metrics, win_rate, elo_gain)

        iteration_time = time.time() - iteration_start
        logger.info(f"Iterazione {iteration} completata in {iteration_time / 60:.1f} minuti")

    # Calcola e registra il tempo totale di training
    total_time = time.time() - start_time
    hours = total_time // 3600
    minutes = (total_time % 3600) // 60

    logger.info(f"Training pipeline completata in {hours:.0f} ore e {minutes:.0f} minuti.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Errore durante l'esecuzione: {str(e)}", exc_info=True)
