import chess
import numpy as np

def index_to_move(idx):
    """
    Converte un indice in una mossa.
    Questa funzione è l'inversa di move_to_index.
    """
    # Verifico che l'indice sia valido
    delta_file = 0
    delta_rank = 0
    if idx is None:
        logger.warning("index_to_move ha ricevuto un indice None")
        return None

    if idx < 0 or idx >= 4672:
        logger.warning(f"index_to_move ha ricevuto un indice fuori range: {idx}")
        return None

    # Gestione promozioni
    if idx >= 3584:
        promotion_idx = idx - 3584
        promotion_type = promotion_idx // 24 + 1  # 1=queen, 2=rook, 3=bishop, 4=knight
        direction = (promotion_idx % 24) // 8  # 0=avanti, 1=destra, 2=sinistra
        file = promotion_idx % 8

        # Determino il pezzo promosso
        promotion_piece = {1: chess.QUEEN, 2: chess.ROOK, 3: chess.BISHOP, 4: chess.KNIGHT}[promotion_type]

        # Calcolo la casella di partenza (pedone in settima/seconda)
        from_rank = 6  # Settima riga (per i bianchi, pedone che sta per promuovere)
        from_file = file

        # Calcolo la casella di arrivo
        to_rank = 7  # Ottava riga (promozione)
        if direction == 0:  # Avanti
            to_file = from_file
        elif direction == 1:  # Destra
            to_file = from_file + 1
        else:  # Sinistra
            to_file = from_file - 1

        # Creo la mossa
        from_square = chess.square(from_file, from_rank)
        to_square = chess.square(to_file, to_rank)

        move = chess.Move(from_square, to_square, promotion=promotion_piece)

    else:
        # Mosse normali
        square_idx = idx // 56
        piece_direction_distance = idx % 56

        # Calcolo file e rank di partenza
        from_file = square_idx % 8
        from_rank = square_idx // 8

        # Calcolo le caratteristiche del pezzo e della direzione
        if piece_direction_distance < 16:  # Pedone bianco (avanti)
            piece_type = 0
            direction = piece_direction_distance // 8
            distance = piece_direction_distance % 8
        elif piece_direction_distance < 32:  # Pedone nero (indietro)
            piece_type = 1
            direction = (piece_direction_distance - 16) // 8
            distance = (piece_direction_distance - 16) % 8
        elif piece_direction_distance < 48:  # Cavallo
            piece_type = 2
            direction = (piece_direction_distance - 32) // 2
            distance = (piece_direction_distance - 32) % 2
        elif piece_direction_distance < 80:  # Alfiere
            piece_type = 3
            direction = (piece_direction_distance - 48) // 8
            distance = (piece_direction_distance - 48) % 8
        elif piece_direction_distance < 112:  # Torre
            piece_type = 4
            direction = (piece_direction_distance - 80) // 8
            distance = (piece_direction_distance - 80) % 8
        elif piece_direction_distance < 176:  # Regina
            piece_type = 5
            direction = (piece_direction_distance - 112) // 8
            distance = (piece_direction_distance - 112) % 8
        else:  # Re
            piece_type = 6
            if piece_direction_distance >= 184:  # Arrocco
                direction = 8 + (piece_direction_distance - 184)  # 8=corto, 9=lungo
            else:
                direction = (piece_direction_distance - 176) // 1
            distance = 0

        # Calcolo le coordinate di arrivo in base al tipo di pezzo e direzione
        if piece_type == 0:  # Pedone bianco
            if direction == 0:  # Avanti
                delta_file = 0
                delta_rank = 1 + distance
            elif direction == 1:  # Avanti destra
                delta_file = 1
                delta_rank = 1
            else:  # Avanti sinistra
                delta_file = -1
                delta_rank = 1
        elif piece_type == 1:  # Pedone nero
            if direction == 0:  # Indietro
                delta_file = 0
                delta_rank = -1 - distance
            elif direction == 1:  # Indietro destra
                delta_file = 1
                delta_rank = -1
            else:  # Indietro sinistra
                delta_file = -1
                delta_rank = -1
        elif piece_type == 2:  # Cavallo
            knight_directions = [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)]
            delta_file, delta_rank = knight_directions[direction]
        elif piece_type == 3:  # Alfiere
            bishop_directions = [(1, 1), (1, -1), (-1, -1), (-1, 1)]
            delta_file, delta_rank = bishop_directions[direction]
            delta_file *= (distance + 1)
            delta_rank *= (distance + 1)
        elif piece_type == 4:  # Torre
            rook_directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            delta_file, delta_rank = rook_directions[direction]
            delta_file *= (distance + 1)
            delta_rank *= (distance + 1)
        elif piece_type == 5:  # Regina
            queen_directions = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]
            delta_file, delta_rank = queen_directions[direction]
            delta_file *= (distance + 1)
            delta_rank *= (distance + 1)
        elif piece_type == 6:  # Re
            if direction >= 8:  # Arrocco
                delta_file = 2 if direction == 8 else -2
                delta_rank = 0
            else:
                king_directions = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]
                delta_file, delta_rank = king_directions[direction]

        # Calcolo le coordinate finali
        to_file = from_file + delta_file
        to_rank = from_rank + delta_rank

        # Creazione della mossa
        from_square = chess.square(from_file, from_rank)
        to_square = chess.square(to_file, to_rank)

        # Verifica validità delle coordinate
        if to_file < 0 or to_file > 7 or to_rank < 0 or to_rank > 7:
            logger.warning(f"index_to_move ha prodotto coordinate non valide per l'indice {idx}")
            return None

        move = chess.Move(from_square, to_square)

    # Verifica finale
    if move is None:
        logger.warning(f"index_to_move ha prodotto una mossa None per l'indice {idx}")

    return move



def policy_index_to_move(policy, board, temperature=0.0):
    """
    Seleziona una mossa dalla distribuzione di policy, con temperatura opzionale.
    """
    # Ottieni le mosse legali
    legal_moves = list(board.legal_moves)

    # Mappa ogni mossa legale al suo indice e ottieni la probabilità
    move_probs = []
    for move in legal_moves:
        try:
            idx = move_to_index(move)
            if 0 <= idx < len(policy):
                move_probs.append((move, policy[idx]))
            else:
                move_probs.append((move, 0))
        except ValueError:
            # Per mosse che non possono essere mappate
            move_probs.append((move, 0))

    # Se non ci sono mosse valide, restituisci None
    if not move_probs:
        return None

    # Normalizza le probabilità
    total_prob = sum(prob for _, prob in move_probs)
    if total_prob > 0:
        move_probs = [(move, prob / total_prob) for move, prob in move_probs]
    else:
        # Se tutte le probabilità sono zero, usa distribuzione uniforme
        move_probs = [(move, 1.0 / len(move_probs)) for move, _ in move_probs]

    # Applica la temperatura
    if temperature > 0:
        # Con temperatura: più alta = più casuale
        adjusted_probs = [(move, prob ** (1.0 / temperature)) for move, prob in move_probs]
        total_adjusted = sum(prob for _, prob in adjusted_probs)
        if total_adjusted > 0:
            adjusted_probs = [(move, prob / total_adjusted) for move, prob in adjusted_probs]

        # Scegli la mossa in base alle probabilità
        choice = random.random()
        cumulative = 0
        for move, prob in adjusted_probs:
            cumulative += prob
            if choice <= cumulative:
                return move
        return adjusted_probs[-1][0]  # Fallback alla mossa con la più alta probabilità
    else:
        # Temperatura zero: scegli semplicemente la mossa con probabilità più alta
        return max(move_probs, key=lambda x: x[1])[0]


def encode_board(board):
    """
    Versione ottimizzata con caching della struttura tensore
    """
    # Struttura tensore: (8, 8, 12)
    board_tensor = np.zeros((8, 8, 12), dtype=np.float32)

    # Mappa di conversione pezzo->canale
    piece_to_channel = {
        (chess.PAWN, chess.WHITE): 0,
        (chess.KNIGHT, chess.WHITE): 1,
        (chess.BISHOP, chess.WHITE): 2,
        (chess.ROOK, chess.WHITE): 3,
        (chess.QUEEN, chess.WHITE): 4,
        (chess.KING, chess.WHITE): 5,
        (chess.PAWN, chess.BLACK): 6,
        (chess.KNIGHT, chess.BLACK): 7,
        (chess.BISHOP, chess.BLACK): 8,
        (chess.ROOK, chess.BLACK): 9,
        (chess.QUEEN, chess.BLACK): 10,
        (chess.KING, chess.BLACK): 11,
    }

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            rank = 7 - chess.square_rank(square)
            file = chess.square_file(square)
            channel = piece_to_channel.get((piece.piece_type, piece.color))
            if channel is not None:
                board_tensor[rank, file, channel] = 1.0

    return board_tensor


def evaluate_terminal(board):
    """
    Valuta il risultato di una posizione terminale.
    Restituisce:
    * 1 se vince il bianco
    * -1 se vince il nero
    * 0 se è patta
    """
    if board.is_checkmate():
        return -1 if board.turn else 1
    else:  # Stallo, ripetizione, regola delle 50 mosse, materiale insufficiente
        return 0


def move_to_index(move: chess.Move) -> int:
    """
    Converte una mossa in un indice univoco (0-1967) per la rete neurale.
    Versione ottimizzata e corretta che garantisce sempre un indice valido.
    """
    # 1. Gestione promozioni (range: 1792-1887 per bianco, 1888-1983 per nero)
    if move.promotion:
        promo_offset = {
            chess.QUEEN: 0,
            chess.ROOK: 1,
            chess.BISHOP: 2,
            chess.KNIGHT: 3
        }[move.promotion]

        from_rank = chess.square_rank(move.from_square)
        from_file = chess.square_file(move.from_square)
        to_file = chess.square_file(move.to_square)

        # Calcola la direzione (-1=sinistra, 0=dritto, 1=destra)
        direction = to_file - from_file

        if from_rank == 6:  # Pedone bianco sulla 7a traversa
            return (1792 + from_file * 12 + (direction + 1) * 4 + promo_offset) % 1968

        elif from_rank == 1:  # Pedone nero sulla 2a traversa
            return (1888 + from_file * 12 + (direction + 1) * 4 + promo_offset) % 1968

        else:  # Fallback per promozioni anomale
            return (1792 + promo_offset) % 1968

    # 2. Gestione arrocco (range: 1950-1953)
    if move in [
        chess.Move.from_uci("e1g1"), chess.Move.from_uci("e1c1"),
        chess.Move.from_uci("e8g8"), chess.Move.from_uci("e8c8")
    ]:
        return {
            chess.Move.from_uci("e1g1"): 1950,  # Arrocco corto bianco
            chess.Move.from_uci("e1c1"): 1951,  # Arrocco lungo bianco
            chess.Move.from_uci("e8g8"): 1952,  # Arrocco corto nero
            chess.Move.from_uci("e8c8"): 1953  # Arrocco lungo nero
        }[move]

    # 3. Calcolo per mosse normali
    from_sq = move.from_square
    to_sq = move.to_square

    # Coordinate e differenze
    from_file, from_rank = chess.square_file(from_sq), chess.square_rank(from_sq)
    to_file, to_rank = chess.square_file(to_sq), chess.square_rank(to_sq)
    delta_file, delta_rank = to_file - from_file, to_rank - from_rank

    # 4. Mappatura direzioni compatta
    direction_map = {
        # Pedoni
        (0, 1): 0, (1, 1): 1, (-1, 1): 2, (0, 2): 3,  # Bianco
        (0, -1): 4, (1, -1): 5, (-1, -1): 6, (0, -2): 7,  # Nero

        # Cavallo
        (1, 2): 8, (2, 1): 9, (2, -1): 10, (1, -2): 11,
        (-1, -2): 12, (-2, -1): 13, (-2, 1): 14, (-1, 2): 15,

        # Direzioni lineari
        (1, 0): 16, (1, 1): 17, (0, 1): 18, (-1, 1): 19,
        (-1, 0): 20, (-1, -1): 21, (0, -1): 22, (1, -1): 23
    }

    # 5. Identifica direzione con fallback
    if (delta_file, delta_rank) in direction_map:
        direction_idx = direction_map[(delta_file, delta_rank)]
    else:
        # Normalizza per mosse lunghe
        norm_file = 0 if delta_file == 0 else delta_file // abs(delta_file)
        norm_rank = 0 if delta_rank == 0 else delta_rank // abs(delta_rank)
        direction_idx = direction_map.get((norm_file, norm_rank), 16)  # Default: est

    # 6. Calcola distanza (solo per mosse lineari)
    distance = max(abs(delta_file), abs(delta_rank))
    if direction_idx >= 16:  # Mosse lineari
        distance = max(0, distance - 1)
    else:
        distance = 0

    # 7. Calcola offset basato sul tipo di mossa
    if direction_idx <= 3:  # Pedone bianco
        piece_offset = 0
    elif direction_idx <= 7:  # Pedone nero
        piece_offset = 24
    elif direction_idx <= 15:  # Cavallo
        piece_offset = 48
    else:  # Altri pezzi
        piece_offset = 56

    # 8. Calcola indice finale (garantito 0-1967)
    square_idx = from_rank * 8 + from_file
    return (square_idx * 73 + piece_offset + direction_idx * 7 + distance) % 1968