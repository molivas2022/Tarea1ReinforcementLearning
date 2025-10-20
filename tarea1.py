"""
Requisitos: Python 3.11
            Idealmente hacer la instalación con Conda
"""

import numpy as np
import gym
import gym_gridworld
import time

# Parametros de ejecución
DEBUG_ENV = False       # Mensaje inicial al cargar el entorno
DEBUG_EPOCH = False     # Mensaje al completar un episodio
RENDER = False          # Abrir ventana al final
WAIT_METHOD = "NONE"    # Cuando se ejecuta playgames(), que hacer entre pasos
                        # INPUT: input(), TIME: esperar 0.2ms, NONE: no hacer nada (pass)

# Función auxiliar para definir que hacer entre pasos cuando se llama playgames()
def wait():
    if WAIT_METHOD == "INPUT": pause = input()
    elif WAIT_METHOD == "TIME": pause = time.sleep(0.2)
    elif WAIT_METHOD == "NONE": pass
    else: raise Exception()

PLOT_Y_MIN = None   # Donde comienza el eje y en el gráfico
PLOT_Y_MAX = None   # Donde termina el eje y en el gráfico

# Función para hacer gráficos
# Requiera matplotlib: si no esta instalado simplemente no se crea la función
try:
    import matplotlib
    matplotlib.use("TkAgg")     # bugfix con la terminal
    import matplotlib.pyplot as plt
    def graph(title, rewards, wins):
        fig, ax = plt.subplots()
        
        ax.set_title(title)
        ax.set_xlabel("episode")
        ax.set_ylabel("reward")

        x = np.arange(len(rewards))
        y = np.array(rewards)

        ax.plot(x, y, label="Rewards")

        # regresión lineal
        coeffs = np.polyfit(x, y, 1)
        trend = np.poly1d(coeffs)
        ax.plot(x, trend(x), "--", label="Trend line")

        fig.show()
        if PLOT_Y_MIN != None or PLOT_Y_MAX != None:
            ax.set_ylim(None if PLOT_Y_MIN == None else PLOT_Y_MIN, None if PLOT_Y_MAX == None else PLOT_Y_MAX)
        plt.pause(0.001)        # bugfix con la terminal
except ImportError:
    def graph(title, rewards):
        pass

"""
Q Learning
Revisar argumentos en train_and_test
"""
def qlearning(env, episodes, max_steps, gamma, initial_lr, final_lr, initial_epsilon, final_epsilon):
    epsilon = initial_epsilon
    lr = initial_lr

    STATES =  env.n_states      # Cantidad de estados en el ambiente.
    ACTIONS = env.n_actions     # Cantidad de acciones posibles.

    Q = np.zeros((STATES, ACTIONS))     # Inicializa la Q table con 0s.
    rewards = []
    for episode in range(episodes):
        rewards_epi=0
        state = env.reset()     # Reinicia el ambiente
        for actual_step in range(max_steps):

            if np.random.uniform(0, 1) < epsilon:   # Escoge un valor al azar entre 0 y 1. Si es menor al valor de epsilon, escoge una acción al azar.
                action = env.action_space.sample() 
            else:
                action = np.argmax(Q[state, :])     # De lo contrario, escogerá el estado con el mayor valor.

            next_state, reward, done, _ = env.step(action) # Ejecuta la acción en el ambiente y guarda los nuevos parámetros (estado siguiente, recompensa, ¿terminó?).
            rewards_epi=rewards_epi+reward

            Q[state, action] = Q[state, action] + lr * (reward + gamma * np.max(Q[next_state, :]) - Q[state, action]) # Calcula la nueva Q table.

            state = next_state

            if DEBUG_EPOCH:
                print(f"Episode {episode} rewards: {rewards_epi}")
                print(f"Value of epsilon: {epsilon}") 

            if (max_steps-2)<actual_step:
                epsilon += (final_epsilon - initial_epsilon) / episodes # Epsilon decay se ajusta al número de episodios
                lr += (final_lr - initial_lr) / episodes                # Alpha decay

            if done:
                epsilon += (final_epsilon - initial_epsilon) / episodes
                lr += (final_lr - initial_lr) / episodes
                break 

        rewards.append(rewards_epi) # Guarda las recompensas en una lista

    return Q, rewards

"""
SARSA
Revisar argumentos en train_and_test
"""
def sarsa(env, episodes, max_steps, gamma, initial_lr, final_lr, initial_epsilon, final_epsilon):
    epsilon = initial_epsilon
    lr = initial_lr

    rewards = []
    STATES =  env.n_states      # Cantidad de estados en el ambiente.
    ACTIONS = env.n_actions     # Cantidad de acciones posibles
    
    Q = np.zeros((STATES, ACTIONS))
    for episode in range(episodes):
        rewards_epi=0
        state = env.reset()     # Reinicia el ambiente
        
        if np.random.uniform(0, 1) < epsilon:   # Escoge un valor al azar entre 0 y 1. Si es menor al valor de epsilon, escoge una acción al azar.
            action = env.action_space.sample() 
        else:
            action = np.argmax(Q[state, :])     # De lo contrario, escogerá el estado con el mayor valor.

        for actual_step in range(max_steps):

            next_state, reward, done, _ = env.step(action)
            
            if np.random.uniform(0, 1) < epsilon:   # Escoge un valor al azar entre 0 y 1. Si es menor al valor de epsilon, escoge una acción al azar.
                action2 = env.action_space.sample() 
            else:
                action2 = np.argmax(Q[next_state, :])   # De lo contrario, escogerá el estado con el mayor valor.

            Q[state, action] = Q[state, action] + lr * (reward + gamma * Q[next_state, action2] - Q[state, action]) #Calcula la nueva Q table.
            rewards_epi=rewards_epi+reward
            state = next_state
            action = action2

            if DEBUG_EPOCH:
                print(f"Episode {episode} rewards: {rewards_epi}")
                print(f"Value of epsilon: {epsilon}") 

            if (max_steps-2)<actual_step:
                epsilon += (final_epsilon - initial_epsilon) / episodes # Epsilon decay se ajusta al número de episodios
                lr += (final_lr - initial_lr) / episodes                # Alpha decay

            if done:
                epsilon += (final_epsilon - initial_epsilon) / episodes
                lr += (final_lr - initial_lr) / episodes
                break
        
        rewards.append(rewards_epi) # Guarda las recompensas en una lista

    return Q, rewards

"""
Double Q Learning
Revisar argumentos en train_and_test
"""
def double_qlearning(env, episodes, max_steps, gamma, initial_lr, final_lr, initial_epsilon, final_epsilon):
    epsilon = initial_epsilon
    lr = initial_lr

    STATES = env.n_states       # Cantidad de estados en el ambiente.
    ACTIONS = env.n_actions     # Cantidad de acciones posibles.

    Q1 = np.zeros((STATES, ACTIONS))  # Inicializa la primera Q table con 0s.
    Q2 = np.zeros((STATES, ACTIONS))  # Inicializa la segunda Q table con 0s.
    rewards = []
    for episode in range(episodes):
        rewards_epi = 0
        state = env.reset()
        for actual_step in range(max_steps):

            # Selección de acción (usando la suma de ambas tablas)
            # Da lo mismo como se seleccione mientras se incorporen las dos tablas, creo.
            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q1[state, :] + Q2[state, :])

            next_state, reward, done, _ = env.step(action)
            rewards_epi += reward

            # Aleatoriamente decide cuál Q table actualizar
            if np.random.rand() < 0.5:
                # Actualiza Q1 usando Q2
                best_action = np.argmax(Q1[next_state, :])
                Q1[state, action] = Q1[state, action] + lr * (
                    reward + gamma * Q2[next_state, best_action] - Q1[state, action]
                )
            else:
                # Actualiza Q2 usando Q1
                best_action = np.argmax(Q2[next_state, :])
                Q2[state, action] = Q2[state, action] + lr * (
                    reward + gamma * Q1[next_state, best_action] - Q2[state, action]
                )

            state = next_state

            if DEBUG_EPOCH:
                print(f"Episode {episode} rewards: {rewards_epi}")
                print(f"Value of epsilon: {epsilon}")

            if (max_steps - 2) < actual_step:
                epsilon += (final_epsilon - initial_epsilon) / episodes # Epsilon decay se ajusta al número de episodios
                lr += (final_lr - initial_lr) / episodes                # Alpha decay

            if done:
                epsilon += (final_epsilon - initial_epsilon) / episodes
                lr += (final_lr - initial_lr) / episodes
                break

        rewards.append(rewards_epi)

    # Q final como la suma promedio de ambas
    Q_final = (Q1 + Q2) / 2.0

    return Q_final, rewards

"""
SARSA(λ)
Revisar argumentos en train_and_test
"""
def sarsa_lambda(lambda_, env, episodes, max_steps, gamma, initial_lr, final_lr, initial_epsilon, final_epsilon):
    epsilon = initial_epsilon
    lr = initial_lr

    STATES = env.n_states       # Cantidad de estados
    ACTIONS = env.n_actions     # Cantidad de acciones

    Q = np.zeros((STATES, ACTIONS))  # Q vacia
    rewards = []
    for episode in range(episodes):
        rewards_epi = 0
        state = env.reset()

        E = np.zeros((STATES, ACTIONS)) # Inicializa las trazas vacias

        # On policy e-greedy: TODO, no repetir codigo
        if np.random.uniform(0, 1) < epsilon:   # exploración
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state, :])     # explotación

        for actual_step in range(max_steps):
            next_state, reward, done, _ = env.step(action)
            rewards_epi += reward

            # On policy e-greedy
            if np.random.uniform(0, 1) < epsilon:   # exploración
                next_action = env.action_space.sample()
            else:
                next_action = np.argmax(Q[next_state, :])   # explotación

            delta = reward + gamma * Q[next_state, next_action] - Q[state, action]  # Calcula el error TD
            E[state, action] += 1   # Para el estado,accion actual se parte sumando 1

            Q += lr * delta * E     # Incorpora el error TD junto a su traza
            E *= gamma * lambda_    # Actualiza todas las trazas

            state = next_state
            action = next_action

            if DEBUG_EPOCH:
                print(f"Episode {episode} rewards: {rewards_epi}")
                print(f"Value of epsilon: {epsilon}")

            if (max_steps - 2) < actual_step:
                epsilon += (final_epsilon - initial_epsilon) / episodes # Epsilon decay se ajusta al número de episodios
                lr += (final_lr - initial_lr) / episodes                # Alpha decay

            if done:
                epsilon += (final_epsilon - initial_epsilon) / episodes
                lr += (final_lr - initial_lr) / episodes
                break
        
        rewards.append(rewards_epi)

    return Q, rewards

"""
Implementación de Watkins de Q(λ)
Revisar argumentos en train_and_test
"""
def qlearning_lambda(lambda_, env, episodes, max_steps, gamma, initial_lr, final_lr, initial_epsilon, final_epsilon):
    epsilon = initial_epsilon
    lr = initial_lr

    STATES = env.n_states       # Cantidad de estados
    ACTIONS = env.n_actions     # Cantidad de acciones

    Q = np.zeros((STATES, ACTIONS)) # Q vacia
    rewards = []
    for episode in range(episodes):
        rewards_epi = 0
        state = env.reset()
        
        E = np.zeros((STATES, ACTIONS)) # Inicializa las trazas vacias

        for actual_step in range(max_steps):
            # Off policy e-greedy
            if np.random.uniform(0, 1) < epsilon:   # exploración
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state, :]) # explotación

            next_state, reward, done, _ = env.step(action)
            rewards_epi += reward

            best_next_action = np.argmax(Q[next_state, :]) # Sesgo de maximización
            delta = reward + gamma * Q[next_state, best_next_action] - Q[state, action] # Calcula el error TD
            E[state, action] += 1 # Para el estado,accion actual se parte sumando 1
            
            Q += lr * delta * E # Incorpora el error TD junto a su traza

            if np.random.uniform(0, 1) < epsilon:   # para despues
                next_action_to_take = env.action_space.sample()
            else:
                next_action_to_take = np.argmax(Q[next_state, :])

            # Watkins: En caso de exploración, reinicar trazas. En caso de explotación, actualizar trazas.
            if next_action_to_take == best_next_action: # explotación
                E *= gamma * lambda_
            else:
                E = np.zeros((STATES, ACTIONS)) # exploración

            state = next_state
            
            if DEBUG_EPOCH:
                print(f"Episode {episode} rewards: {rewards_epi}")
                print(f"Value of epsilon: {epsilon}")

            if (max_steps - 2) < actual_step:
                epsilon += (final_epsilon - initial_epsilon) / episodes # Epsilon decay se ajusta al número de episodios
                lr += (final_lr - initial_lr) / episodes                # Alpha decay

            if done:
                epsilon += (final_epsilon - initial_epsilon) / episodes
                lr += (final_lr - initial_lr) / episodes
                break
                
        rewards.append(rewards_epi)

    return Q, rewards

"""
Función para correr juegos siguiendo una determinada política
Es decir, para el testeo 
"""
def playgames(env, Q, num_games, max_steps, render = True):
    rewards = []
    wins = 0
    env.reset()
    if render: env.render()

    for i_episode in range(num_games):
        rewards_epi=0
        observation = env.reset()
        t = 0
        while t < max_steps:    # Hay que definir un limite
            action = np.argmax(Q[observation, :])   # La acción a realizar esta dada por la política
            observation, reward, done, info = env.step(action)
            rewards_epi=rewards_epi+reward
            if render: env.render()
            wait()  # Función custom para decidir que separa cada paso
            if done:
                if reward >= 0:
                    wins += 1
                break
            t += 1
        rewards.append(rewards_epi)
    wait()
    env.close()
    return wins, rewards

"""
Dado un:
- Algoritmo
- Entorno
- Hiperparámetros
Función para ejecutar todo el ciclo de:
1. Aprender la politica usando ese algoritmo (Train)
2. Evaluar el resultado final varias veces (Test)
3. Gráficar
"""
def train_and_test(title = "title",         # nombre del gráfico
                   env = "deterministic",   # determinista o estocastico
                   algorithm = "sarsa",     # el algoritmo para aprender la politica
                   lambda_ = 0.5,           # solo en caso de ser un algoritmo TD(λ)
                   map = "map1.txt",        # el mapa del entorno
                   episodes = 10000,        # episodios de entrenamiento
                   max_steps = 100,         # pasos máximos de un episodio (train o test)
                   gamma = 0.90,            # factor de descuento
                   initial_lr = 0.2,        # valor inicial de lr que decae linealmente
                   final_lr = 0.2,          # valor en el ultimo episodio de lr que decae linealmente
                   initial_epsilon = 1,     # valor inicial de epsilon que decae linealmente
                   final_epsilon = 0.1):    # valor en el ultimo episodio de epsilon que decae linealmente
    
    if env == "deterministic":
        env = gym.make("GridWorld-v0", file_name=map)
    elif env == "random":
        env = gym.make("GridWorldRandom-v0", file_name=map)
    else: raise Exception()
    env.verbose = True
    _ =env.reset()

    if DEBUG_ENV:
        if hasattr(env, 'file_name'): print('Map path:', env.file_name)
        print('Observation space:', env.observation_space)
        print('Action space:', env.action_space)
    
    if algorithm == "sarsa":
        Q, train_rewards = sarsa(env, episodes, max_steps, gamma, initial_lr, final_lr, initial_epsilon, final_epsilon)
    elif algorithm == "qlearning":
        Q, train_rewards = qlearning(env, episodes, max_steps, gamma, initial_lr, final_lr, initial_epsilon, final_epsilon)
    elif algorithm == "double_qlearning":
        Q, train_rewards = double_qlearning(env, episodes, max_steps, gamma, initial_lr, final_lr, initial_epsilon, final_epsilon)
    elif algorithm == "sarsa_lambda":
        Q, train_rewards = sarsa_lambda(lambda_, env, episodes, max_steps, gamma, initial_lr, final_lr, initial_epsilon, final_epsilon)
    elif algorithm == "qlearning_lambda":
        Q, train_rewards = qlearning_lambda(lambda_, env, episodes, max_steps, gamma, initial_lr, final_lr, initial_epsilon, final_epsilon)
    else: raise Exception()

    test_wins, test_rewards = playgames(env, Q, 100, max_steps, RENDER)
    env.close()

    graph(title, train_rewards, test_wins)

    print(title + ":", test_wins/len(train_rewards))


### -- EXPERIMENTOS -- ###

def question1():
    global PLOT_Y_MIN
    global PLOT_Y_MAX
    PLOT_Y_MIN = -10
    PLOT_Y_MAX = 4

    train_and_test(title="Sarsa: Map 1", algorithm="sarsa")   
    train_and_test(title="Q Learning: Map 1", algorithm="qlearning")

    train_and_test(title="Sarsa: Map 2", algorithm="sarsa", map="map2.txt")
    train_and_test(title="Q Learning: Map 2", algorithm="qlearning", map="map2.txt")

    hyperparams = {"episodes": 20000, "max_steps": 500, "gamma": 0.99, "initial_lr": 0.02, "final_lr": 0.02}
    train_and_test(title="Sarsa - gamma = 0.99 - learning rate = 0.02: Map 2", algorithm="sarsa", map="map2.txt", **hyperparams)

    hyperparams = {"episodes": 20000, "max_steps": 500, "gamma": 0.99, "initial_lr": 0.2, "final_lr": 0.02}
    train_and_test(title="Q Learning - gamma = 0.99 - decaying learning rate: Map 2", algorithm="qlearning", map="map2.txt", **hyperparams)

def question2():
    global PLOT_Y_MIN
    global PLOT_Y_MAX
    PLOT_Y_MIN = -10
    PLOT_Y_MAX = 4

    hyperparams = {"episodes": 20000, "max_steps": 500, "gamma": 0.99, "initial_lr": 0.02, "final_lr": 0.02}
    train_and_test(title="Sarsa: Map 2", algorithm="sarsa", map="map2.txt", **hyperparams)
    train_and_test(title="Sarsa: Map 2 (Stochastic)", algorithm="sarsa", map="map2.txt", env="random", **hyperparams)

    hyperparams = {"episodes": 20000, "max_steps": 500, "gamma": 0.99, "initial_lr": 0.2, "final_lr": 0.02}
    train_and_test(title="Q Learning: Map 2", algorithm="qlearning", map="map2.txt", **hyperparams)
    train_and_test(title="Q Learning: Map 2 (Stochastic)", algorithm="qlearning", map="map2.txt", env="random", **hyperparams)

def question3():
    global PLOT_Y_MIN
    global PLOT_Y_MAX
    PLOT_Y_MIN = -10
    PLOT_Y_MAX = 4

    hyperparams = {"episodes": 20000, "max_steps": 500, "gamma": 0.99, "initial_lr": 0.2, "final_lr": 0.02}

    train_and_test(title="Q Learning: Map 2 (Stochastic)", algorithm="qlearning", map="map2.txt", env="random", **hyperparams)
    train_and_test(title="Double Q Learning: Map 2 (Stochastic)", algorithm="double_qlearning", map="map2.txt", env="random", **hyperparams)

def question4():
    global PLOT_Y_MIN
    global PLOT_Y_MAX
    PLOT_Y_MIN = -10
    PLOT_Y_MAX = 4

    hyperparams = {"episodes": 20000, "max_steps": 500, "gamma": 0.99, "initial_lr": 0.02, "final_lr": 0.02}
    train_and_test(title="Sarsa: Map 2 (Stochastic)", algorithm="sarsa", map="map2.txt", env="random", **hyperparams)
    train_and_test(title="Sarsa(λ): Map 2 (Stochastic)", algorithm="sarsa_lambda", map="map2.txt", env="random", **hyperparams)

    hyperparams = {"episodes": 20000, "max_steps": 500, "gamma": 0.99, "initial_lr": 0.2, "final_lr": 0.02}
    train_and_test(title="Q Learning: Map 2 (Stochastic)", algorithm="qlearning", map="map2.txt", env="random", **hyperparams)
    train_and_test(title="Watkins's Q(λ): Map 2 (Stochastic)", algorithm="qlearning_lambda", map="map2.txt", env="random", **hyperparams)

question1()
question2()
question3()
question4()
input("Press ENTER to close program")