import numpy as np
import chess
import logging
from tqdm import tqdm
import concurrent.futures
import time
import copy
import os
from mcts import MCTS
from src.core.utils import move_to_index


# Configurazione del logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('evaluator')


def play_single_game(game_idx, new_mcts, old_mcts, temperature, max_moves=200):
    if game_idx % 2 == 0:
        white_mcts, black_mcts = new_mcts, old_mcts
        white_name, black_name = "Nuovo", "Vecchio"
    else:
        white_mcts, black_mcts = old_mcts, new_mcts
        white_name, black_name = "Vecchio", "Nuovo"

    board = chess.Board()
    move_count = 0

    while not board.is_game_over(claim_draw=True):
        current_mcts = white_mcts if board.turn == chess.WHITE else black_mcts
        current_mcts.update_state(board.copy())

        try:
            action_probs = current_mcts.get_action_probs(board, temperature=temperature)
            valid_moves = list(board.legal_moves)

            if not valid_moves:
                break

            move_indices = [move_to_index(m) for m in valid_moves]
            best_idx = np.argmax([action_probs[idx] for idx in move_indices])
            board.push(valid_moves[best_idx])
            move_count += 1

        except Exception as e:
            logger.error(f"Errore partita {game_idx + 1}: {e}")
            break

    result = board.result()

    if result == "1-0":
        if white_name == "Nuovo":
            logger.info(f"Partita {game_idx + 1}: Nuovo modello (Bianco) vince")
            return 1.0
        else:
            logger.info(f"Partita {game_idx + 1}: Vecchio modello (Bianco) vince")
            return 0.0

    elif result == "0-1":
        if black_name == "Nuovo":
            logger.info(f"Partita {game_idx + 1}: Nuovo modello (Nero) vince")
            return 1.0
        else:
            logger.info(f"Partita {game_idx + 1}: Vecchio modello (Nero) vince")
            return 0.0

    else:
        logger.info(f"Partita {game_idx + 1}: Patta")
        return 0.5

def evaluate_model(new_model, old_model, num_games=20, mcts_simulations=100, temperature=0.0):
    """
    Valuta un nuovo modello contro un modello precedente attraverso partite di confronto.

    Args:
        new_model: Il nuovo modello da valutare
        old_model: Il modello precedente usato come baseline
        num_games: Numero di partite da giocare
        mcts_simulations: Numero di simulazioni MCTS per mossa
        temperature: Temperatura per la selezione delle mosse

    Returns:
        Tasso di vittoria del nuovo modello (0.0-1.0)
    """
    # Inizializza MCTS di base
    base_new_mcts = MCTS(model=new_model, num_simulations=mcts_simulations)
    base_old_mcts = MCTS(model=old_model, num_simulations=mcts_simulations)

    logger.info(f"Iniziando la valutazione: {num_games} partite con {mcts_simulations} simulazioni MCTS per mossa")

    # Configura parallelismo
    num_workers = min(3, max(1, os.cpu_count() - 1)) # Max 3 thread, lasciando un core libero
    results = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Avvia tutte le partite in parallelo
        futures = [
            executor.submit(
                play_single_game,
                game_idx,
                copy.deepcopy(base_new_mcts), # Ogni thread ha la sua copia
                copy.deepcopy(base_old_mcts),
                temperature
            )
            for game_idx in range(num_games)
        ]

        # Monitora il progresso e raccogli risultati
        with tqdm(total=num_games, desc="SImulando partite") as pbar:
            for future in concurrent.futures.as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    logger.error(f"Errore nella simulazione: {e}")
                    results.append(0.5) # Considera patta in caso di errore
                pbar.update(1)

        # Calcola statistiche finali
        wins = sum(r == 1.0 for r in results)
        draws = sum(r == 0.5 for r in results)
        losses = sum(r == 0.0 for r in results)

        win_rate = (wins + 0.5 * draws) / num_games

        logger.info(f"Valutazione completata: {wins} vittorie, {draws} patte, {losses} sconfitte")
        logger.info(f"Tasso di vittoria del nuovo modello: {win_rate:.2%}")

        return win_rate


def evaluate_opening_book(model, opening_positions, mcts_simulations=100):
    """
    Valuta un modello su una serie di posizioni di apertura conosciute.

    Args:
        model: Modello da valutare
        opening_positions: Elenco di posizioni di apertura in formato FEN
        mcts_simulations: Numero di simulazioni MCTS per posizione

    Returns:
        Dizionario con statistiche sulla valutazione delle aperture
    """
    mcts = MCTS(model=model, num_simulations=mcts_simulations)
    results = {}

    for opening_name, fen in tqdm(opening_positions.items(), desc="Aperture"):
        board = chess.Board(fen)

        # Ottieni la policy e il valore per questa posizione
        action_probs = mcts.get_action_probs(board, temperature=0.1)
        value = mcts.search(board).value()

        # Trova le mosse migliori
        valid_moves = list(board.legal_moves)
        if not valid_moves:
            continue

        move_indices = [move_to_index(move) for move in valid_moves]
        move_probs = [(move, action_probs[idx]) for move, idx in zip(valid_moves, move_indices)]
        move_probs.sort(key=lambda x: x[1], reverse=True)

        # Salva i risultati
        top_moves = [{'move': str(move), 'probability': float(prob)} for move, prob in move_probs[:3]]

        results[opening_name] = {
            'value': float(value),
            'top_moves': top_moves
        }

    return results


def evaluate_with_elo_system(models, num_games_per_pair=10, base_elo=1500):
    """
    Valuta una serie di modelli usando il sistema ELO per il rating.

    Args:
        models: Dizionario con {nome_modello: modello}
        num_games_per_pair: Numero di partite da giocare per ogni coppia di modelli
        base_elo: Rating ELO iniziale per tutti i modelli

    Returns:
        Dizionario con i rating ELO finali per ogni modello
    """
    # Inizializza i rating ELO
    elo_ratings = {name: base_elo for name in models.keys()}

    # Funzione per aggiornare il rating ELO
    def update_elo(rating_a, rating_b, score_a, k=32):
        """Aggiorna il rating ELO in base al risultato della partita."""
        expected_a = 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
        new_rating_a = rating_a + k * (score_a - expected_a)
        return new_rating_a

    # Gioca partite tra tutte le coppie di modelli
    model_names = list(models.keys())

    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            model_a_name, model_b_name = model_names[i], model_names[j]
            model_a, model_b = models[model_a_name], models[model_b_name]

            logger.info(f"Valutazione: {model_a_name} vs {model_b_name}")

            # Fai giocare i modelli tra loro
            win_rate = evaluate_model(model_a, model_b, num_games=num_games_per_pair)

            # Aggiorna i rating ELO
            score_a = win_rate  # Punteggio del modello A (0-1)
            score_b = 1 - win_rate  # Punteggio del modello B

            old_rating_a = elo_ratings[model_a_name]
            old_rating_b = elo_ratings[model_b_name]

            elo_ratings[model_a_name] = update_elo(old_rating_a, old_rating_b, score_a)
            elo_ratings[model_b_name] = update_elo(old_rating_b, old_rating_a, score_b)

            logger.info(f"Risultato: {model_a_name} ({elo_ratings[model_a_name]:.0f}) - "
                        f"{model_b_name} ({elo_ratings[model_b_name]:.0f})")

    # Ordina i rating in ordine decrescente
    sorted_ratings = sorted(elo_ratings.items(), key=lambda x: x[1], reverse=True)

    logger.info("\n=== CLASSIFICA ELO ===")
    for rank, (name, rating) in enumerate(sorted_ratings, 1):
        logger.info(f"{rank}. {name}: {rating:.0f}")

    return dict(sorted_ratings)


def analyze_positions(model, positions, depth=3):
    """
    Analizza una serie di posizioni usando MCTS con diversi livelli di profondità.

    Args:
        model: Modello da utilizzare
        positions: Elenco di posizioni in formato FEN
        depth: Profondità massima di analisi

    Returns:
        Analisi dettagliata delle posizioni
    """
    results = {}

    for i, fen in enumerate(positions):
        board = chess.Board(fen)
        position_result = {
            'fen': fen,
            'legal_moves': len(list(board.legal_moves)),
            'analysis': {}
        }

        # Analizza la posizione con diversi livelli di simulazioni MCTS
        for sim_count in [50, 200, 800]:
            mcts = MCTS(model=model, num_simulations=sim_count)

            start_time = time.time()
            root = mcts.search(board)
            elapsed = time.time() - start_time

            # Ottieni le migliori mosse
            children = sorted(root.children.items(), key=lambda x: x[1].visit_count, reverse=True)
            top_moves = []

            for move, child in children[:5]:
                top_moves.append({
                    'move': str(move),
                    'visits': child.visit_count,
                    'value': float(child.value()),
                    'prior': float(child.prior)
                })

            position_result['analysis'][sim_count] = {
                'top_moves': top_moves,
                'root_value': float(root.value()),
                'time_seconds': elapsed
            }

        results[f"position_{i + 1}"] = position_result

    return results


if __name__ == "__main__":
    # Test di valutazione
    from self_play import create_neural_network_model

    # Crea due modelli identici per il test
    model_a = create_neural_network_model()
    model_b = create_neural_network_model()

    # Valuta i modelli l'uno contro l'altro (dovrebbe essere circa 0.5)
    win_rate = evaluate_model(model_a, model_b, num_games=10, mcts_simulations=100)

    print(f"Test di valutazione completato. Tasso di vittoria: {win_rate:.2f}")