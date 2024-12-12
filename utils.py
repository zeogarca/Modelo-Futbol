import matplotlib.pyplot as plt

def plot_training_rewards(rewards):
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, label="Recompensas")
    plt.xlabel("Episodios")
    plt.ylabel("Recompensa")
    plt.title("Progreso de Entrenamiento")
    plt.legend()
    plt.show()