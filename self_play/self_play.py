import numpy as np
import tensorflow as tf
import chess
import os
import time
import random
import logging
from tqdm import tqdm

# Configurazione del logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('self_play')

from mcts import MCTS
from src.core.utils import move_to_index, index_to_move, policy_index_to_move, encode_board, evaluate_terminal


class SelfPlayGenerator:
    """
    Genera partite attraverso self-play utilizzando MCTS e un modello di rete neurale.
    """

    def __init__(self, model, num_games=100, mcts_simulations=50,
                 temperature=1.0, temperature_drop_move=10,
                 dirichlet_alpha=0.3, dirichlet_weight=0.25, c_puct=1.0):
        """
        Inizializza il generatore di self-play.

        Args:
            model: Modello di rete neurale che fornisce policy e value
            num_games: Numero di partite da generare
            mcts_simulations: Numero di simulazioni MCTS per mossa
            temperature: Parametro di temperatura per la selezione delle mosse
            temperature_drop_move: Mossa dopo la quale ridurre la temperatura a 0
            dirichlet_alpha: Parametro alpha per il rumore Dirichlet
            dirichlet_weight: Peso del rumore Dirichlet rispetto alla policy della rete
            c_puct: Parametro di esplorazione nell'UCB
        """
        self.model = model
        self.num_games = num_games
        self.mcts_simulations = mcts_simulations
        self.temperature = temperature
        self.temperature_drop_move = temperature_drop_move
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_weight = dirichlet_weight
        self.c_puct = c_puct

        # Inizializza MCTS con il modello
        self.mcts = MCTS(
            model=self.model,
            num_simulations=self.mcts_simulations,
            dirichlet_noise=True,  # Attiva il rumore Dirichlet per favorire l'esplorazione
            dirichlet_alpha= self.dirichlet_alpha,
            dirichlet_weight= self.dirichlet_weight,
            c_puct= self.c_puct
        )

    def generate_games(self):
        """
        Genera partite di scacchi attraverso self-play.

        Returns:
            Lista di dati di partite, ciascuna con una lista di stati, politiche e risultato.
        """
        games_data = []
        max_moves = 75

        for game_idx in tqdm(range(self.num_games), desc="Generazione partite"):
            # Inizializza la partita
            board = chess.Board()
            states = []
            policies = []
            current_player = 1  # 1 per il bianco, -1 per il nero

            # L'MCTS iterativo si resetta a ogni mossa, quindi non è necessario un reset esplicito
            # self.mcts.reset()

            move_count = 0
            logger.info(f"Iniziando partita {game_idx + 1}/{self.num_games}")

            # Gioca fino alla fine della partita
            while not board.is_game_over(claim_draw=True):
                # Determina la temperatura corrente
                current_temp = self.temperature if move_count < self.temperature_drop_move else 0

                # Ottieni la canonical form della board (dal punto di vista del giocatore corrente)
                canonical_board = encode_board(board)
                states.append(canonical_board)

                # Ottieni le probabilità dell'azione da MCTS come vettore
                action_probs_vector = self.mcts.get_action_probs(board, temperature=current_temp)
                policies.append(action_probs_vector) # Salva il vettore direttamente

                # Seleziona una mossa in base alle probabilità
                valid_moves = list(board.legal_moves)
                if not valid_moves:
                    break # Nessuna mossa legale, la partita è finita

                # Mappa le probabilità alle mosse valide
                move_indices = [move_to_index(m) for m in valid_moves]
                valid_probs = np.array([action_probs_vector[idx] for idx in move_indices])

                # Normalizza le probabilità per assicurare che sommino a 1
                if np.sum(valid_probs) > 0:
                    valid_probs /= np.sum(valid_probs)
                else:
                    # Se tutte le probabilità sono zero, usa una distribuzione uniforme
                    valid_probs = np.ones(len(valid_moves)) / len(valid_moves)

                # Campiona una mossa in base alle probabilità
                selected_idx = np.random.choice(len(valid_moves), p=valid_probs)
                move = valid_moves[selected_idx]

                # Esegui la mossa
                board.push(move)
                move_count += 1

                # Cambia giocatore
                current_player = -current_player

            # Determina il risultato della partita
            result = evaluate_terminal(board)

            # Salva i dati della partita
            game_data = {
                'states': states,
                'policies': policies,
                'result': result
            }
            games_data.append(game_data)

            logger.info(f"Partita {game_idx + 1} completata con risultato: {result}")

        import gc

        gc.collect()

        return games_data

def create_neural_network_model(input_shape=(8, 8, 12)):
    """
    Crea un modello di rete neurale per scacchi con architettura simile ad AlphaZero.

    Args:
        input_shape: Forma dell'input (default: (8, 8, 12) per la rappresentazione della scacchiera)

    Returns:
        Modello TensorFlow/Keras
    """
    inputs = tf.keras.Input(shape=input_shape)

    # Rete comune
    x = tf.keras.layers.Conv2D(256, 3, padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    # Blocchi residuali
    for _ in range(3):  # AlphaZero usa 19 blocchi residuali
        res_input = x
        x = tf.keras.layers.Conv2D(64, 3, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Conv2D(256, 3, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.add([res_input, x])
        x = tf.keras.layers.ReLU()(x)

    # Policy head
    policy_head = tf.keras.layers.Conv2D(2, 1)(x)
    policy_head = tf.keras.layers.BatchNormalization()(policy_head)
    policy_head = tf.keras.layers.ReLU()(policy_head)
    policy_head = tf.keras.layers.Flatten()(policy_head)
    policy_head = tf.keras.layers.Dense(1968, activation='softmax', name='policy')(
        policy_head)  # 1968 possibili mosse in scacchi

    # Value head
    value_head = tf.keras.layers.Conv2D(1, 1)(x)
    value_head = tf.keras.layers.BatchNormalization()(value_head)
    value_head = tf.keras.layers.ReLU()(value_head)
    value_head = tf.keras.layers.Flatten()(value_head)
    value_head = tf.keras.layers.Dense(256, activation='relu')(value_head)
    value_head = tf.keras.layers.Dense(1, activation='tanh', name='value')(value_head)

    # Creazione del modello
    model = tf.keras.Model(inputs=inputs, outputs=[policy_head, value_head])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss={
            'policy': 'categorical_crossentropy',
            'value': 'mean_squared_error'
        }
    )

    return model

# Se eseguito direttamente, esegui un test di self-play
if __name__ == "__main__":
    def test_self_play(model_path=None, num_games=1): # Funzione di test per il self-play
        """
        Testa il processo di self-play con un modello esistente o un nuovo modello.

        Args:
            model_path: Path al modello da caricare (se None, crea un nuovo modello)
            num_games: Numero di partite da generare
        """
        if model_path and os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
            logger.info(f"Modello caricato da {model_path}")
        else:
            model = create_neural_network_model()
            logger.info("Nuovo modello creato")

        generator = SelfPlayGenerator(
            model=model,
            num_games=num_games,
            mcts_simulations=100,  # Ridotto per il test
            temperature=1.0
        )

        games_data = generator.generate_games()
        logger.info(f"Generate {len(games_data)} partite")

        # Per debug: mostra un esempio di stato e politica dalla prima partita
        if games_data:
            game = games_data[0]
            logger.info(f"Esempio - Numero di stati: {len(game['states'])}")
            logger.info(f"Esempio - Forma stato: {game['states'][0].shape}")
            logger.info(f"Esempio - Forma policy: {game['policies'][0].shape}")
            logger.info(f"Esempio - Risultato: {game['result']}")

        return games_data
