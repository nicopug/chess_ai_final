import numpy as np
import chess
import tensorflow as tf
import logging
from src.core.utils import encode_board, move_to_index

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('MCTS')

class Node:
    """Nodo nell'albero di Monte Carlo Search Tree."""
    def __init__(self, board, parent=None, prior=0.0, move=None):
        self.board = board
        self.parent = parent
        self.prior = prior
        self.children = {}  # Mossa -> Nodo figlio
        self.visit_count = 0
        self.value_sum = 0.0
        self.expanded = False
        self.move = move

    def value(self):
        """Valore Q del nodo (prospettiva del giocatore che ha mosso per arrivare qui)."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def expand(self, policy):
        """Espande il nodo creando figli per ogni mossa legale."""
        if self.expanded:
            return
        self.expanded = True
        for move in self.board.legal_moves:
            try:
                move_idx = move_to_index(move)
                prior = policy[move_idx] if 0 <= move_idx < len(policy) else 1e-6
                new_board = self.board.copy()
                new_board.push(move)
                self.children[move] = Node(new_board, self, prior, move)
            except Exception as e:
                logger.error(f"Errore nell'espansione della mossa {move}: {e}")

    def select_child(self, c_puct):
        """Seleziona il figlio con il punteggio UCB più alto."""
        best_score = -np.inf
        best_move = None
        best_child = None
        for move, child in self.children.items():
            # Il valore del figlio è dal suo punto di vista, quindi lo neghiamo per ottenere il Q-value
            q_value = -child.value() if child.visit_count > 0 else 0
            exploration = c_puct * child.prior * np.sqrt(self.visit_count) / (1 + child.visit_count)
            score = q_value + exploration
            if score > best_score:
                best_score = score
                best_move = move
                best_child = child
        return best_move, best_child

class MCTS:
    """Implementazione iterativa e robusta di MCTS."""
    def __init__(self, model, num_simulations=100, c_puct=1.0,
                 dirichlet_noise=False, dirichlet_alpha=0.3, dirichlet_weight=0.1):
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.dirichlet_noise = dirichlet_noise
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_weight = dirichlet_weight
        self.root = None

    def get_action_probs(self, state, temperature=1.0):
        """Esegue le simulazioni MCTS e restituisce le probabilità delle mosse."""
        self.root = Node(state.copy())
        self._expand_and_evaluate(self.root) # Espandi sempre la radice
        if self.dirichlet_noise:
            self._add_dirichlet_noise()

        for _ in range(self.num_simulations):
            self._search()

        visits = {move: child.visit_count for move, child in self.root.children.items()}
        if not visits:
            return np.zeros(1968)

        counts = np.array(list(visits.values()))
        if temperature == 0:
            best_move_index = np.argmax(counts)
            best_move = list(visits.keys())[best_move_index]
            probs = {move: (1.0 if move == best_move else 0.0) for move in visits.keys()}
        else:
            counts_temp = counts ** (1.0 / temperature)
            total = np.sum(counts_temp)
            probs = {move: count / total for move, count in zip(visits.keys(), counts_temp)}

        return self._probs_to_vector(probs)

    def _search(self):
        """Esegue una singola simulazione (selezione, espansione, backup)."""
        path = [self.root]
        
        # 1. Selezione
        while path[-1].expanded:
            current_node = path[-1]
            if current_node.board.is_game_over():
                break
            move, child_node = current_node.select_child(self.c_puct)
            if child_node is None:
                break
            path.append(child_node)

        leaf_node = path[-1]
        
        # 2. Espansione e Valutazione
        if leaf_node.board.is_game_over():
            value = self._handle_terminal_state(leaf_node.board)
        else:
            value = self._expand_and_evaluate(leaf_node)

        # 3. Backup
        for node in reversed(path):
            node.visit_count += 1
            node.value_sum += value
            value = -value # Inverti il valore per il genitore

    def _expand_and_evaluate(self, node):
        """Espande un nodo e restituisce la valutazione della rete neurale."""
        board_tensor = encode_board(node.board)
        board_tensor = tf.expand_dims(tf.convert_to_tensor(board_tensor), 0)
        policy, value = self.model(board_tensor, training=False)
        policy = policy.numpy().flatten()
        value = value.numpy()[0, 0]
        node.expand(policy)
        return value

    def _add_dirichlet_noise(self):
        """Aggiunge rumore di Dirichlet ai priori della radice."""
        if self.root.expanded and len(self.root.children) > 0:
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(self.root.children))
            for i, child in enumerate(self.root.children.values()):
                child.prior = child.prior * (1 - self.dirichlet_weight) + noise[i] * self.dirichlet_weight

    def _handle_terminal_state(self, board):
        """Restituisce il valore dal punto di vista del giocatore corrente."""
        if board.is_checkmate():
            return -1.0 # Se è il mio turno e sono in scacco matto, ho perso
        return 0.0 # Patta

    def _probs_to_vector(self, probs_dict):
        """Converte un dizionario di probabilità di mosse in un vettore."""
        vector = np.zeros(1968)
        for move, prob in probs_dict.items():
            try:
                idx = move_to_index(move)
                if 0 <= idx < 1968:
                    vector[idx] = prob
            except Exception as e:
                logger.error(f"Conversione mossa fallita in probs_to_vector: {e}")
        return vector