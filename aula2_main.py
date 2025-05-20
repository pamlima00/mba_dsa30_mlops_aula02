import mlflow
from simtoseis_library import carregar_dados, calcular_residuos, plot_simulation_distribution
import numpy as np

# Parâmetro externo
proporcao_treino = 0.75

# Configurar para usar o servidor local
mlflow.set_tracking_uri("http://localhost:5000")

# Início do experimento
with mlflow.start_run():

    # Log do parâmetro
    mlflow.log_param("proporcao_treino", proporcao_treino)

    # Carregamento dos dados
    sim_slice, seismic_slice, seismic_gt, seis_estimado = carregar_dados()

    # Plot (apenas visual)
    plot_simulation_distribution(sim_slice, bins=35, titulo="Distribuição do Dado Simulado")

    # Calcular e salvar resíduos
    residuos = calcular_residuos(seis_estimado, seismic_gt)
    np.save("residuos.npy", residuos)

    # Logar como artefato
    mlflow.log_artifact("residuos.npy")

    # Logar métrica
    rmse = np.sqrt(np.mean(residuos ** 2))
    mlflow.log_metric("rmse", rmse)
