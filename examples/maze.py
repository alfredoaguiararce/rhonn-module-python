import streamlit as st
import random

# Definir el laberinto como una matriz con emojis
maze = [
    ['🟦', '🟦', '🟦', '🟦', '🟦', '🟦', '🟦', '🟦', '🟦'],
    ['🟦', '🚶‍♂️', ' ', ' ', '🟦', ' ', ' ', ' ', '🟦'],
    ['🟦', '🟦', '🟦', ' ', '🟦', ' ', '🟦', ' ', '🟦'],
    ['🟦', ' ', ' ', ' ', ' ', ' ', ' ', ' ', '🟦'],
    ['🟦', ' ', '🟦', '🟦', '🟦', '🟦', '🟦', '🟦', '🟦'],
    ['🟦', ' ', ' ', ' ', ' ', ' ', ' ', ' ', '🟦'],
    ['🟦', ' ', '🟦', '🟦', '🟦', ' ', '🟦', ' ', '🟦'],
    ['🟦', ' ', ' ', ' ', ' ', ' ', ' ', '🏁', '🟦'],
    ['🟦', '🟦', '🟦', '🟦', '🟦', '🟦', '🟦', '🟦', '🟦']
]

# Definir las acciones posibles del agente como índices (arriba, abajo, izquierda, derecha)
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
actions = [UP, DOWN, LEFT, RIGHT]

# Crear controles en el panel lateral
st.sidebar.title("Configuración de Q-Learning")
learning_rate = st.sidebar.slider("Tasa de Aprendizaje", 0.0, 1.0, 0.1, step=0.01)
discount_factor = st.sidebar.slider("Factor de Descuento", 0.0, 1.0, 0.9, step=0.01)
exploration_prob = st.sidebar.slider("Probabilidad de Exploración", 0.0, 1.0, 0.2, step=0.01)
num_episodes = st.sidebar.slider("Número de Episodios", 1, 5000, 1000)

# Función para verificar si una posición es válida en el laberinto
def is_valid(x, y):
    return 0 <= x < len(maze) and 0 <= y < len(maze[0]) and maze[x][y] != '🟦'

# Función para realizar el aprendizaje Q-learning
def q_learning(maze, learning_rate, discount_factor, exploration_prob, num_episodes):
    q_table = {}
    for episode in range(num_episodes):
        x, y = 1, 1  # Iniciar en el punto de inicio (🚶‍♂️)
        while maze[x][y] != '🏁':
            if random.uniform(0, 1) < exploration_prob:
                action = random.choice(actions)
            else:
                action = max(actions, key=lambda a: q_table.get(((x, y), a), 0))

            dx, dy = 0, 0
            if action == UP:
                dx, dy = -1, 0
            elif action == DOWN:
                dx, dy = 1, 0
            elif action == LEFT:
                dx, dy = 0, -1
            elif action == RIGHT:
                dx, dy = 0, 1

            new_x, new_y = x + dx, y + dy

            if is_valid(new_x, new_y):
                next_max_q = max(q_table.get(((new_x, new_y), a), 0) for a in actions)
                reward = -1 if maze[new_x][new_y] != '🏁' else 100  # Recompensa positiva al llegar a la meta
                q_table[((x, y), action)] = q_table.get(((x, y), action), 0) + \
                                             learning_rate * (reward + discount_factor * next_max_q - q_table.get(((x, y), action), 0))
                x, y = new_x, new_y

    return q_table

# Función para seguir la política óptima aprendida por el agente
def follow_policy(q_table, maze):
    x, y = 1, 1
    path = []
    while maze[x][y] != '🏁':
        action = max(actions, key=lambda a: q_table.get(((x, y), a), 0))
        dx, dy = 0, 0
        if action == UP:
            dx, dy = -1, 0
        elif action == DOWN:
            dx, dy = 1, 0
        elif action == LEFT:
            dx, dy = 0, -1
        elif action == RIGHT:
            dx, dy = 0, 1
        new_x, new_y = x + dx, y + dy
        path.append((x, y))
        x, y = new_x, new_y
    path.append((x, y))
    return path

# Crear la aplicación Streamlit
st.title("Simulación de Q-Learning en un Laberinto")

# Entrenar al agente Q-learning con los valores seleccionados en el panel lateral
q_table = q_learning(maze, learning_rate, discount_factor, exploration_prob, num_episodes)

# Crear una tabla para mostrar el laberinto
maze_table = []
for row in maze:
    maze_table.append(['' if cell != '🟦' else '🟦' for cell in row])

# Marcar el camino óptimo en la tabla del laberinto
optimal_path = follow_policy(q_table, maze)
for x, y in optimal_path:
    maze_table[x][y] = '🚀'

# Mostrar el laberinto con el camino óptimo
st.markdown("## Laberinto con Camino Óptimo")
st.table(maze_table)