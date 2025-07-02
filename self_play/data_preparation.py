import numpy as np
import random


def prepare_training_data(games_data):
    """
    Prepara i dati delle partite per il training della rete neurale.

    Args:
        games_data: Lista di dizionari, ciascuno contenente stati, politiche e risultato di una partita

    Returns:
        Dizionario con stati, politiche target e valori target pronti per il training
    """
    states = []
    policy_targets = []
    value_targets = []

    # Cicla su ogni partita
    for game in games_data:
        game_states = game['states']
        game_policies = game['policies']
        game_result = game['result']

        # Cicla su ogni posizione nella partita
        for idx, (state, policy) in enumerate(zip(game_states, game_policies)):
            # Aggiungi lo stato
            states.append(state)

            # Aggiungi la policy target
            policy_targets.append(policy)

            # Calcola il valore target in base al risultato finale
            # Nota: il valore è dal punto di vista del giocatore attuale
            # Se idx è pari, è il turno del bianco; se dispari, è il turno del nero
            player_to_move = 1 if idx % 2 == 0 else -1
            value_target = game_result * player_to_move

            value_targets.append(value_target)

    # Converte in array numpy
    states = np.array(states)
    policy_targets = np.array(policy_targets)
    value_targets = np.array(value_targets).reshape(-1, 1)

    # Applica data augmentation per la simmetria della scacchiera (DISABILITATO)
    # augmented_data = apply_symmetry_augmentation(states, policy_targets, value_targets)
    # return augmented_data

    # Mescola i dati senza augmentation
    indices = np.arange(len(states))
    np.random.shuffle(indices)

    return {
        'states': states[indices],
        'policy_targets': policy_targets[indices],
        'value_targets': value_targets[indices]
    }


def apply_symmetry_augmentation(states, policy_targets, value_targets):
    """
    Applica l'augmentation basata sulle simmetrie della scacchiera.
    Nel caso degli scacchi, possiamo usare la simmetria orizzontale (flip).

    Args:
        states: Array degli stati della scacchiera
        policy_targets: Array delle policy target
        value_targets: Array dei valori target

    Returns:
        Dizionario con i dati aumentati
    """
    # Crea copie per il flipping
    flipped_states = np.flip(states, axis=2).copy()

    # Per le policy, devi mappare le mosse in base alla nuova disposizione della scacchiera
    # Nota: questa è una semplificazione, una vera implementazione richiederebbe
    # una mappatura dettagliata delle mosse in base alla simmetria
    # Per ora, supponiamo che la policy sia una matrice che può essere flippata direttamente
    flipped_policy_targets = np.flip(policy_targets.reshape(-1, 8, 8, 73), axis=1).reshape(-1, 4672)

    # I valori target rimangono invariati
    flipped_value_targets = value_targets.copy()

    # Concatena i dati originali e quelli aumentati
    augmented_states = np.concatenate([states, flipped_states])
    augmented_policy_targets = np.concatenate([policy_targets, flipped_policy_targets])
    augmented_value_targets = np.concatenate([value_targets, flipped_value_targets])

    # Mescola i dati
    indices = np.arange(len(augmented_states))
    np.random.shuffle(indices)

    return {
        'states': augmented_states[indices],
        'policy_targets': augmented_policy_targets[indices],
        'value_targets': augmented_value_targets[indices]
    }


def balance_training_data(training_data, max_samples=100000):
    """
    Bilancia i dati di training per evitare che il modello favorisca un certo risultato.

    Args:
        training_data: Dizionario con gli stati, le policy target e i valori target
        max_samples: Numero massimo di campioni da mantenere

    Returns:
        Dizionario con i dati bilanciati
    """
    states = training_data['states']
    policy_targets = training_data['policy_targets']
    value_targets = training_data['value_targets']

    # Dividi i dati in base al risultato
    win_indices = np.where(value_targets > 0.3)[0]
    draw_indices = np.where(np.abs(value_targets) <= 0.3)[0]
    loss_indices = np.where(value_targets < -0.3)[0]

    # Calcola quanti campioni prendere da ciascuna categoria
    total_samples = min(len(states), max_samples)
    samples_per_category = total_samples // 3

    # Seleziona casualmente i campioni da ciascuna categoria
    selected_win = np.random.choice(win_indices, min(samples_per_category, len(win_indices)), replace=False)
    selected_draw = np.random.choice(draw_indices, min(samples_per_category, len(draw_indices)), replace=False)
    selected_loss = np.random.choice(loss_indices, min(samples_per_category, len(loss_indices)), replace=False)

    # Unisci gli indici selezionati
    selected_indices = np.concatenate([selected_win, selected_draw, selected_loss])
    np.random.shuffle(selected_indices)

    # Crea il nuovo dataset bilanciato
    balanced_data = {
        'states': states[selected_indices],
        'policy_targets': policy_targets[selected_indices],
        'value_targets': value_targets[selected_indices]
    }

    return balanced_data


def load_training_data(file_paths):
    """
    Carica i dati di training da file salvati.

    Args:
        file_paths: Lista di percorsi dei file .npz contenenti i dati delle partite

    Returns:
        Dizionario con i dati di training combinati
    """
    all_states = []
    all_policies = []
    all_results = []

    for file_path in file_paths:
        data = np.load(file_path, allow_pickle=True)

        # Estrai i dati
        game_states = data['states']
        game_policies = data['policies']
        game_results = data['results']

        # Raccogli gli stati, le policy e i risultati
        for i in range(len(game_states)):
            states = game_states[i]
            policies = game_policies[i]
            result = game_results[i]

            for j in range(len(states)):
                all_states.append(states[j])
                all_policies.append(policies[j])

                # Il valore dipende dal turno del giocatore
                player_to_move = 1 if j % 2 == 0 else -1
                value = result * player_to_move
                all_results.append(value)

    # Converti in array numpy
    all_states = np.array(all_states)
    all_policies = np.array(all_policies)
    all_results = np.array(all_results).reshape(-1, 1)

    return {
        'states': all_states,
        'policy_targets': all_policies,
        'value_targets': all_results
    }


if __name__ == "__main__":
    # Test per verificare il funzionamento
    import numpy as np

    # Crea dati di test
    test_games = []
    for _ in range(3):
        test_game = {
            'states': [np.random.rand(8, 8, 12) for _ in range(10)],
            'policies': [np.random.rand(1968) for _ in range(10)],
            'result': np.random.choice([-1, 0, 1])
        }
        test_games.append(test_game)

    # Prepara i dati
    training_data = prepare_training_data(test_games)

    # Verifica le dimensioni
    print(f"Stati: {training_data['states'].shape}")
    print(f"Policy targets: {training_data['policy_targets'].shape}")
    print(f"Value targets: {training_data['value_targets'].shape}")
