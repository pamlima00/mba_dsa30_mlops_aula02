import numpy as np
import matplotlib.pyplot as plt

def carregar_dados():
    sim_slice = np.load("sim_slice.npy")
    seismic_slice = np.load("seismic_slice.npy")
    seismic_gt = np.load("seismic_slice_GT.npy")
    seis_estimado = np.load("seis_estimado.npy")
    return sim_slice, seismic_slice, seismic_gt, seis_estimado

def calcular_residuos(estimado, real):
    return real - estimado

def plot_simulation_distribution(sim_slice, bins=30, titulo="Histograma dos Dados"):
    plt.hist(sim_slice[:, -1], bins=bins)
    plt.title(titulo)
    plt.xlabel("Valor")
    plt.ylabel("Frequência")
    plt.grid(True)
    plt.show()
