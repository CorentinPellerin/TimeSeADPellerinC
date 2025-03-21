import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import torch.jit
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import MSELoss
import torch.optim as optim
import timesead
import matplotlib.widgets as widgets
import itertools
import time
from torch.utils.data import DataLoader, TensorDataset
from timesead.optim.trainer import Trainer, default_log_fn, EarlyStoppingHook, CheckpointHook
from timesead.optim.loss import TorchLossWrapper, LogCoshLoss
from timesead.evaluation.evaluator import Evaluator
from timesead.evaluation.ts_precision_recall import ts_precision_and_recall, compute_window_indices
from timesead.models.reconstruction.lstm_ae import LSTMAEAnomalyDetector, LSTMAEMirza2018, LSTMAEMalhotra2016
from timesead.models.reconstruction.tcn_ae import TCNAEAnomalyDetector, TCNAE
from timesead.models.generative.omni_anomaly import OmniAnomaly, OmniAnomalyDetector, OmniAnomalyLoss
from timesead.models.prediction.tcn_prediction import TCNPrediction, TCNPredictionAnomalyDetector, TCNS2SPrediction, TCNS2SPredictionAnomalyDetector
from timesead.models.prediction.lstm_prediction import LSTMPrediction, LSTMS2SPrediction, LSTMPredictionAnomalyDetector, LSTMS2SPredictionAnomalyDetector
from timesead.models.generative.lstm_vae_gan import LSTMVAEGAN, LSTMVAEGANAnomalyDetector

#FONCTION TNCP
def collate_fn_tcn(batch):
    """
    Construit les batches pour TCNPrediction.
    Chaque entrée est de forme (B, T, D) et la cible correspond au dernier pas de temps : (B, D)
    """
    xs, ys = zip(*batch)
    xs = torch.stack(xs)  # (B, T, D)
    ys = torch.stack(ys)  # (B, D)
    return xs, ys
    
#TEST TCNP 
class TCNPredictionTester:
    """
    Cette classe permet de :
      1. Optimiser les hyperparamètres d'un modèle TCNPrediction via un split train/test.
      2. Tester le modèle sur le jeu de test en calculant la loss (entre la prédiction et la cible).
      3. Visualiser le score d'anomalie (calculé sur le dernier pas de temps de chaque fenêtre de X_test).
    
    On suppose que :
      - Le modèle TCNPrediction est encapsulé dans un TCNPredictionAnomalyDetector.
      - Les tenseurs d'entraînement et de test sont de forme (N, T, D) et que la cible est le dernier pas de temps.
    """
    
    def __init__(self, X_train, X_test):
        """
        Parameters
        ----------
        X_train : torch.Tensor
            Tenseur d'entraînement de forme (N, T, D)
        X_test : torch.Tensor
            Tenseur de test de forme (N, T, D)
        """
        self.X_train = X_train
        self.X_test = X_test
        self.input_dim = X_train.shape[-1]
        self.window_size = X_train.shape[1]  # Longueur de la séquence
    
    def optimize_hyperparameters(self):
        """
        Recherche sur une grille d'hyperparamètres (par exemple, 'filters' et 'kernel_sizes')
        et choisit la configuration qui minimise la loss sur le jeu de test.
        
        Retourne
        -------
        best_model : Le modèle TCNPredictionAnomalyDetector entraîné avec la meilleure configuration.
        best_params : dict des meilleurs hyperparamètres.
        best_loss : float, la loss correspondante.
        """
        param_grid = {
            'filters': [ (32, 32) ], #, (64, 64)
            'kernel_sizes': [ (3, 3) ], #, (5, 3)
            'linear_hidden_layers': [ (50,) ], #, (100, 50)
            'activation': [ torch.nn.ReLU() ], #, torch.nn.LeakyReLU()
            'learning_rate': [1e-3]
        }

        best_loss = float('inf')
        best_params = None
        best_model = None
        collate_fn = collate_fn_tcn  # fonction de collate pour TCN

        # Préparation des DataLoaders pour l'entraînement et le test.
        # La cible correspond au dernier pas de temps de chaque séquence.
        train_dataset = TensorDataset(
            self.X_train, 
            self.X_train[:, -1, :]  # cible : dernier pas de temps
        )
        test_dataset = TensorDataset(
            self.X_test, 
            self.X_test[:, -1, :]
        )
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=32, collate_fn=collate_fn)
    
        for params in itertools.product(*param_grid.values()):
            param_dict = dict(zip(param_grid.keys(), params))
            start_time = time.time()

            # Instanciation du modèle TCNPrediction avec la configuration courante.
            # On fixe ici prediction_horizon=1 pour prédire le dernier pas.
            base_model = TCNPrediction(
                input_dim=self.input_dim,
                window_size=self.window_size,
                filters=param_dict['filters'],
                kernel_sizes=param_dict['kernel_sizes'],
                prediction_horizon=1
            )
            detector = TCNPredictionAnomalyDetector(base_model)
            
            criterion = TorchLossWrapper(torch.nn.MSELoss())
            optimizer = torch.optim.Adam(detector.model.parameters(), lr=1e-3)
            
            # Entraînement (par exemple, 5 époques)
            detector.model.train()
            for epoch in range(5):
                for batch, target in train_loader:
                    x = batch  # x de forme (B, T, D)
                    optimizer.zero_grad()
                    # Prédiction : attend (B, 1, D)
                    output = detector.model((x,))
                    if isinstance(output, tuple):
                        output = output[0]
                    loss = criterion(output.squeeze(1), target)
                    loss.backward()
                    optimizer.step()
            
            # Calcul des statistiques sur le jeu d'entraînement (pour le scoring d'anomalie)
            detector.model.eval()
            all_errors = []
            with torch.no_grad():
                for batch, target in train_loader:
                    x = batch
                    output = detector.model((x,))
                    # Calcul de l'erreur entre la cible et la prédiction
                    error = torch.abs(target - output.squeeze(1))  # forme (B, D)
                    all_errors.append(error)
            all_errors = torch.cat(all_errors, dim=0)  # (N, D)
            # Calcul de la moyenne sur le dernier pas (vecteur de forme (D,))
            detector.mean = all_errors.mean(dim=0)
            # Centrer les erreurs et calculer la matrice de covariance
            errors_centered = all_errors - detector.mean.unsqueeze(0)
            cov = torch.matmul(errors_centered.T, errors_centered) / (errors_centered.shape[0] - 1)
            # Pour la stabilité numérique, ajouter une petite valeur sur la diagonale
            cov.diagonal().add_(1e-5)
            detector.precision = torch.inverse(cov)
            
            # Évaluation sur le jeu de test
            detector.model.eval()
            total_loss = 0
            with torch.no_grad():
                for batch, target in test_loader:
                    x = batch
                    output = detector.model((x,))
                    total_loss += criterion(output.squeeze(1), target).item()
            test_loss = total_loss / len(test_loader)
            elapsed_time = time.time() - start_time
            print(f"Test Loss avec {param_dict} : {test_loss:.4f} (Temps: {elapsed_time:.2f} sec)")
            
            if test_loss < best_loss:
                best_loss = test_loss
                best_params = param_dict
                best_model = detector

        print("\n")
        print(f"Meilleurs hyperparamètres: {best_params} avec Loss {best_loss:.4f}")
        self.best_model = best_model
        self.best_params = best_params
        self.best_loss = best_loss
        return best_model, best_params, best_loss

    def test_model(self):
        """
        Teste le modèle optimisé en calculant la loss sur le jeu de test.
        Pour TCNPrediction, la cible est le dernier pas de temps de la fenêtre.
        """
        if not hasattr(self, 'best_model'):
            raise ValueError("Veuillez d'abord optimiser les hyperparamètres.")
        
        detector = self.best_model
        criterion = TorchLossWrapper(torch.nn.MSELoss())
        test_dataset = TensorDataset(
            self.X_test, 
            self.X_test[:, -1, :]
        )
        test_loader = DataLoader(test_dataset, batch_size=32, collate_fn=collate_fn_tcn)
        
        detector.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch, target in test_loader:
                x = batch
                output = detector.model((x,))
                total_loss += criterion(output.squeeze(1), target).item()
        test_loss = total_loss / len(test_loader)
        print(f"Loss sur l'ensemble de test : {test_loss}")
        self.test_loss = test_loss
        return test_loss

    def plot_anomaly_scores(self):
        """
        Calcule et affiche le score d'anomalie pour chaque fenêtre de X_test.
        Pour chaque fenêtre, on utilise l'erreur entre la cible (dernier pas) et la prédiction,
        corrigée par la moyenne et la covariance calculées lors de l'entraînement.
        """
        if not hasattr(self, 'best_model'):
            raise ValueError("Veuillez d'abord optimiser les hyperparamètres.")
        
        detector = self.best_model
        anomaly_scores = []
        # Pour TCNPrediction, la cible est le dernier pas de temps.
        # On itère sur chaque fenêtre de X_test.
        for i in range(len(self.X_test)):
            # Préparer la fenêtre : on conserve la forme (1, T, D)
            window = self.X_test[i].unsqueeze(0)
            with torch.no_grad():
                # Prédiction : (1, 1, D)
                pred = detector.model((window,))
            # Cible : dernier pas de temps de la fenêtre
            target = window[:, -1, :]  # (1, D)
            error = target - pred.squeeze(1)  # (1, D)
            error = error.abs()
            # Calcul du score d'anomalie : on soustrait la moyenne et on applique un produit bilinéaire
            error = error - detector.mean.unsqueeze(0)  # (1, D)
            score = F.bilinear(error, error, detector.precision.unsqueeze(0))
            anomaly_scores.append(score.squeeze().cpu().numpy())
        anomaly_scores = np.array(anomaly_scores)
        
        plt.figure(figsize=(12, 6))
        plt.plot(anomaly_scores, label="Scores d'anomalie (TCN Prediction)", color="blue", alpha=0.7)
        plt.title("Détection d'anomalies avec TCNPrediction")
        plt.xlabel("Index temporel")
        plt.ylabel("Score d'anomalie")
        plt.legend()
        plt.show()

        df_scores = pd.DataFrame({
        "anomaly_score": anomaly_scores,
        "index": np.arange(len(anomaly_scores))
        })

        '''
        mean_score = np.mean(anomaly_scores)
        std_score = np.std(anomaly_scores)
        threshold = mean_score + 3 * std_score
        anomalies = anomaly_scores > threshold
        num_anomalies = np.sum(anomalies)
        ratio_anomalies = num_anomalies / len(anomaly_scores) * 100.0
        
        print(f"Threshold: {threshold:.4f}")
        print(f"Nombre d'anomalies détectées : {num_anomalies}")
        print(f"Ratio d'anomalies : {ratio_anomalies:.2f}%")
        '''

        mean_score = np.mean(anomaly_scores)
        std_score = np.std(anomaly_scores)
        z_scores = (anomaly_scores - mean_score) / std_score
    
        # On fixe un seuil global sur les Z-scores (par exemple, 3)
        threshold = 3.0
        anomalies = z_scores > threshold
        num_anomalies = np.sum(anomalies)
        ratio_anomalies = num_anomalies / len(z_scores) * 100.0
    
        print(f"Seuil (Z-score): {threshold:.4f}")
        print(f"Nombre d'anomalies détectées : {num_anomalies}")
        print(f"Ratio d'anomalies : {ratio_anomalies:.2f}%")

        plt.figure(figsize=(10, 6))
        sns.histplot(df_scores["anomaly_score"], bins=50, kde=True)
        plt.title("Distribution des scores d'anomalie")
        plt.xlabel("Score d'anomalie")
        plt.ylabel("Fréquence")
        plt.show()
        self.anomaly_scores = anomaly_scores
        return anomaly_scores
        
#FONCTIONS TNCS2SP
def collate_fn_tcns2s(batch):
    xs, ys = zip(*batch)
    xs = torch.stack(xs)  # (N, T, D)
    ys = torch.stack(ys)  # (N, T, D)
    return xs, ys
    
def full_sequence_forward_tcns2s(model, x):
    # Ici, on envoie x sous forme de tuple, si le modèle l'exige.
    return model((x,))

#CLASS TEST TCNS2SP
class TCNS2SPredictionTester:
    """
    Cette classe permet de :
      1. Optimiser les hyperparamètres d'un modèle TCNS2SPrediction via un split train/test.
      2. Tester le modèle sur le jeu de test en calculant la loss (entre la reconstruction et la séquence d'entrée).
      3. Visualiser le score d'anomalie (calculé sur le dernier pas de temps de chaque fenêtre de X_test).

    On suppose que :
      - Le modèle TCNS2SPrediction est encapsulé dans un TCNS2SPredictionAnomalyDetector.
      - Les tenseurs d'entraînement et de test sont de forme (N, T, D) et que la cible est la séquence entière,
        de sorte que l'on peut comparer le dernier pas de la reconstruction avec le dernier pas de l'entrée.
    """
    def __init__(self, X_train, X_test):
        """
        Parameters
        ----------
        X_train : torch.Tensor
            Tenseur d'entraînement de forme (N, T, D)
        X_test : torch.Tensor
            Tenseur de test de forme (N, T, D)
        """
        self.X_train = X_train
        self.X_test = X_test
        self.input_dim = X_train.shape[-1]
        self.window_size = X_train.shape[1]  # Longueur de la séquence

    def optimize_hyperparameters(self):
        """
        Recherche sur une grille d'hyperparamètres (par exemple, 'filters', 'kernel_sizes', 'dilations',
        'last_n_layers_to_cat', 'activation' et 'learning_rate') et choisit la configuration qui minimise
        la loss sur le jeu de test.

        Returns
        -------
        best_model : Le modèle TCNS2SPredictionAnomalyDetector entraîné avec la meilleure configuration.
        best_params : dict des meilleurs hyperparamètres.
        best_loss : float, la loss correspondante.
        """
        param_grid = {
            'filters': [ (64, 64, 64, 64, 64), (32, 32, 32, 32, 32) ],
            'kernel_sizes': [ (3, 3, 3, 3, 3) ],
            'dilations': [ (1, 2, 4, 8, 16) ],
            'last_n_layers_to_cat': [3],
            'activation': [torch.nn.ReLU()],
            'learning_rate': [1e-3]
        }
        best_loss = float('inf')
        best_params = None
        best_model = None
        collate_fn = collate_fn_tcns2s  # fonction de collate adaptée

        # Pour TCNS2SPredictionAnomalyDetector, on considère que la cible est la séquence entière
        train_dataset = TensorDataset(self.X_train, self.X_train)
        test_dataset = TensorDataset(self.X_test, self.X_test)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=32, collate_fn=collate_fn)

        for params in itertools.product(*param_grid.values()):
            param_dict = dict(zip(param_grid.keys(), params))
            start_time = time.time()

            # Instanciation du modèle TCNS2SPrediction avec la configuration courante
            base_model = TCNS2SPrediction(
                input_dim=self.input_dim,
                filters=param_dict['filters'],
                kernel_sizes=param_dict['kernel_sizes'],
                dilations=param_dict['dilations'],
                last_n_layers_to_cat=param_dict['last_n_layers_to_cat'],
                activation=param_dict['activation']
            )
            # Pour le calcul des erreurs sur le dernier pas, on fixe offset à 1
            detector = TCNS2SPredictionAnomalyDetector(base_model, offset=1)

            criterion = TorchLossWrapper(torch.nn.MSELoss())
            optimizer = torch.optim.Adam(detector.model.parameters(), lr=param_dict['learning_rate'])

            # Entraînement (par exemple, 5 époques)
            detector.model.train()
            for epoch in range(5):
                for batch, target in train_loader:
                    x = batch  # x de forme (B, T, D)
                    optimizer.zero_grad()
                    output = full_sequence_forward_tcns2s(detector.model, x)
                    loss = criterion(output, x)
                    loss.backward()
                    optimizer.step()

            # Calcul des statistiques sur le jeu d'entraînement (pour le scoring d'anomalie)
            detector.model.eval()
            all_errors = []
            with torch.no_grad():
                for batch, target in train_loader:
                    x = batch
                    output = full_sequence_forward_tcns2s(detector.model, x)
                    # Pour le scoring, on compare le dernier pas de la séquence
                    target_last = x[:, -1, :]    # (B, D)
                    pred_last = output[:, -1, :]   # (B, D)
                    error = torch.abs(target_last - pred_last)
                    all_errors.append(error)
            all_errors = torch.cat(all_errors, dim=0)  # (N_total, D)
            detector.mean = all_errors.mean(dim=0)
            errors_centered = all_errors - detector.mean.unsqueeze(0)
            cov = torch.matmul(errors_centered.T, errors_centered) / (errors_centered.shape[0] - 1)
            cov.diagonal().add_(1e-5)
            detector.precision = torch.inverse(cov)

            # Évaluation sur le jeu de test
            detector.model.eval()
            total_loss = 0
            with torch.no_grad():
                for batch, target in test_loader:
                    x = batch
                    output = full_sequence_forward_tcns2s(detector.model, x)
                    total_loss += criterion(output, x).item()
            test_loss = total_loss / len(test_loader)
            elapsed_time = time.time() - start_time
            print(f"Test Loss avec {param_dict} : {test_loss:.4f} (Temps: {elapsed_time:.2f} sec)")

            if test_loss < best_loss:
                best_loss = test_loss
                best_params = param_dict
                best_model = detector

        print("\nMeilleurs hyperparamètres:", best_params, "avec Loss", best_loss)
        self.best_model = best_model
        self.best_params = best_params
        self.best_loss = best_loss
        return best_model, best_params, best_loss

    def test_model(self):
        """
        Teste le modèle optimisé en calculant la loss sur le jeu de test.
        Pour TCNS2SPrediction, la cible est la séquence entière.
        """
        if not hasattr(self, 'best_model'):
            raise ValueError("Veuillez d'abord optimiser les hyperparamètres.")

        detector = self.best_model
        criterion = TorchLossWrapper(torch.nn.MSELoss())
        test_dataset = TensorDataset(self.X_test, self.X_test)
        test_loader = DataLoader(test_dataset, batch_size=32, collate_fn=collate_fn_tcns2s)

        detector.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch, target in test_loader:
                x = batch
                output = full_sequence_forward_tcns2s(detector.model, x)
                total_loss += criterion(output, x).item()
        test_loss = total_loss / len(test_loader)
        print(f"Loss sur l'ensemble de test : {test_loss}")
        self.test_loss = test_loss
        return test_loss

    def plot_anomaly_scores(self):
        """
        Calcule et affiche le score d'anomalie pour chaque fenêtre de X_test.
        Pour chaque fenêtre, on compare le dernier pas de la séquence de la reconstruction à celui de l'entrée,
        puis on corrige l'erreur avec la moyenne et la covariance calculées lors de l'entraînement.
        """
        if not hasattr(self, 'best_model'):
            raise ValueError("Veuillez d'abord optimiser les hyperparamètres.")

        detector = self.best_model
        anomaly_scores = []
        # Itération sur chaque fenêtre de X_test (chaque échantillon de la dimension N)
        for i in range(len(self.X_test)):
            # Préparer la fenêtre sous la forme (1, T, D)
            window = self.X_test[i].unsqueeze(0)
            with torch.no_grad():
                # Prédiction : (1, T, D)
                pred = detector.model((window,))
            # Cible : dernier pas de temps de la fenêtre
            target = window[:, -1, :]  # (1, D)
            pred_last = pred[:, -1, :]  # (1, D)
            error = torch.abs(target - pred_last)
            # Correction par la moyenne et application du produit bilinéaire
            error = error - detector.mean.unsqueeze(0)
            score = F.bilinear(error, error, detector.precision.unsqueeze(0))
            anomaly_scores.append(score.squeeze().cpu().numpy())
        anomaly_scores = np.array(anomaly_scores).squeeze()

        plt.figure(figsize=(12, 6))
        plt.plot(anomaly_scores, label="Scores d'anomalie (TCNS2S Prediction)", color="blue", alpha=0.7)
        plt.title("Détection d'anomalies avec TCNS2SPrediction")
        plt.xlabel("Index temporel")
        plt.ylabel("Score d'anomalie")
        plt.legend()
        plt.show()

        df_scores = pd.DataFrame({
            "anomaly_score": anomaly_scores,
            "index": np.arange(len(anomaly_scores))
        })

        mean_score = np.mean(anomaly_scores)
        std_score = np.std(anomaly_scores)
        z_scores = (anomaly_scores - mean_score) / std_score

        # Utilisation d'un seuil global sur les Z-scores (par exemple, 3)
        threshold = 3.0
        anomalies = z_scores > threshold
        num_anomalies = np.sum(anomalies)
        ratio_anomalies = num_anomalies / len(z_scores) * 100.0

        print(f"Seuil (Z-score): {threshold:.4f}")
        print(f"Nombre d'anomalies détectées : {num_anomalies}")
        print(f"Ratio d'anomalies : {ratio_anomalies:.2f}%")

        plt.figure(figsize=(10, 6))
        sns.histplot(df_scores["anomaly_score"], bins=50, kde=True)
        plt.title("Distribution des scores d'anomalie")
        plt.xlabel("Score d'anomalie")
        plt.ylabel("Fréquence")
        plt.show()

        self.anomaly_scores = anomaly_scores
        return anomaly_scores

#FONCTIONS LSTMP
def collate_fn_lstmp(batch):
    xs, ys = zip(*batch)
    xs = torch.stack(xs)  # (N, T, D)
    # Pour LSTMPrediction, on transpose pour obtenir (T, N, D)
    xs = xs.transpose(0, 1)
    # Ici, la cible est identique à l'entrée (on souhaite reproduire la séquence)
    ys = xs.clone()
    return xs, ys
    
def collate_fn_lstmp_test(batch):
    xs, _ = zip(*batch)
    xs = torch.stack(xs)        # xs a la forme (batch, T, D)
    xs = xs.transpose(0, 1)      # réorganise pour obtenir (T, batch, D)
    # Récupérer l'horizon de prédiction à partir du modèle
    # Ici, on suppose que detector.model est un SingleInputWrapper qui encapsule le modèle original
    # et que le modèle original est accessible via detector.model.model
    ph = detector.model.model.prediction_horizon if hasattr(detector.model, 'model') else detector.model.prediction_horizon
    # Extraire les derniers ph pas temporels
    ys = xs[-ph:]
    return xs, (ys, ys)

def full_sequence_forward_lstmp(model, x):
    # Envoie l'input sous forme de tuple si le modèle l'exige.
    return model((x,))

#CLASS TEST LSTMP
class LSTMPredictionTester:
    """
    Cette classe permet de :
      1. Optimiser les hyperparamètres d'un modèle LSTMPrediction (Malhotra 2015) via un split train/test.
      2. Tester le modèle sur le jeu de test en calculant la loss (entre la prédiction et la cible).
      3. Visualiser le score d'anomalie (calculé sur le dernier pas de temps de la prédiction).
    
    On suppose que :
      - Le modèle LSTMPrediction est encapsulé dans un LSTMPredictionAnomalyDetector.
      - Les tenseurs d'entraînement et de test sont de forme (N, T, D).
      - Le modèle attend des tenseurs d'entrée de forme (T, B, D) (d'où la transposition dans le collate).
    """
    def __init__(self, X_train, X_test):
        """
        Parameters
        ----------
        X_train : torch.Tensor
            Tenseur d'entraînement de forme (N, T, D)
        X_test : torch.Tensor
            Tenseur de test de forme (N, T, D)
        """
        self.X_train = X_train
        self.X_test = X_test
        self.input_dim = X_train.shape[-1]
        self.window_size = X_train.shape[1]

    def optimize_hyperparameters_manual(self):
        """
        Recherche sur une grille d'hyperparamètres (par exemple, 'lstm_hidden_dims', 'linear_hidden_layers' et 'learning_rate')
        et choisit la configuration qui minimise la loss sur le jeu de test.
        
        Returns
        -------
        best_model : Le modèle LSTMPredictionAnomalyDetector entraîné avec la meilleure configuration.
        best_params : dict des meilleurs hyperparamètres.
        best_loss : float, la loss correspondante.
        """
        param_grid = {
            'lstm_hidden_dims': [[64, 64, 32]], #[30, 20], [50, 30], 
            'linear_hidden_layers': [[100, 50]], #[],[50], 
            'linear_activation': [torch.nn.ELU()], #, torch.nn.ReLU()
            'prediction_horizon': [3], #5
            'learning_rate': [1e-3], #5e-4
            'optimizer': ['adam'] #'sgd', 
        }
        best_loss = float('inf')
        best_params = None
        best_model = None

        # Pour LSTMPrediction, on transpose les données de (N, T, D) à (T, N, D)
        train_dataset = TensorDataset(self.X_train.transpose(0, 1), self.X_train.transpose(0, 1))
        test_dataset  = TensorDataset(self.X_test.transpose(0, 1), self.X_test.transpose(0, 1))
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn_lstmp)
        test_loader  = DataLoader(test_dataset, batch_size=32, collate_fn=collate_fn_lstmp)

        for params in itertools.product(*param_grid.values()):
            param_dict = dict(zip(param_grid.keys(), params))
            start_time = time.time()

            # Instanciation du modèle LSTMPrediction avec prediction_horizon fixé à 3
            base_model = LSTMPrediction(
                input_dim=self.input_dim,
                lstm_hidden_dims=param_dict['lstm_hidden_dims'],
                linear_hidden_layers=param_dict['linear_hidden_layers'],
                prediction_horizon=3
            )
            # Encapsulation dans l'anomaly detector
            detector = LSTMPredictionAnomalyDetector(base_model)

            criterion = LogCoshLoss() #torch.nn.MSELoss()
            optimizer = torch.optim.Adam(detector.model.parameters(), lr=param_dict['learning_rate'])

            # Entraînement (5 époques par exemple)
            detector.model.train()
            for epoch in range(5):
                for batch, target in train_loader:
                    x = batch  # x: (T, B, D)
                    optimizer.zero_grad()
                    output = full_sequence_forward(detector.model, x)  # output: (horizon, B, D)
                    loss = criterion(output, x[-detector.model.prediction_horizon:,:,:])
                    loss.backward()
                    optimizer.step()

            # Calcul des statistiques pour le scoring (sur le dernier pas)
            detector.model.eval()
            all_errors = []
            with torch.no_grad():
                for batch, target in train_loader:
                    x = batch  # (T, B, D)
                    output = full_sequence_forward(detector.model, x)  # (horizon, B, D)
                    # On utilise le dernier pas de prédiction
                    pred_last = output[-1, :, :]  # (B, D)
                    target_last = x[-1, :, :]      # (B, D)
                    error = torch.abs(target_last - pred_last)  # (B, D)
                    all_errors.append(error)
            all_errors = torch.cat(all_errors, dim=0)  # (N_total, D)
            detector.mean = all_errors.mean(dim=0)
            errors_centered = all_errors - detector.mean.unsqueeze(0)
            cov = torch.matmul(errors_centered.T, errors_centered) / (errors_centered.shape[0] - 1)
            cov.diagonal().add_(1e-5)
            detector.precision = torch.inverse(cov)

            # Évaluation sur le jeu de test
            detector.model.eval()
            total_loss = 0
            with torch.no_grad():
                for batch, target in test_loader:
                    x = batch
                    output = full_sequence_forward(detector.model, x)
                    total_loss += criterion(output, x[-detector.model.prediction_horizon:,:,:]).item()
            test_loss = total_loss / len(test_loader)
            elapsed_time = time.time() - start_time
            print(f"Test Loss avec {param_dict} : {test_loss:.4f} (Temps: {elapsed_time:.2f} sec)")

            if test_loss < best_loss:
                best_loss = test_loss
                best_params = param_dict
                best_model = detector

        print("\nMeilleurs hyperparamètres:", best_params, "avec Loss", best_loss)
        self.best_model = best_model
        self.best_params = best_params
        self.best_loss = best_loss
        return best_model, best_params, best_loss
        
    def optimize_hyperparameters_trainer_timesead(self):
        """
        Recherche sur une grille d'hyperparamètres en entraînant le modèle sur self.X_train 
        et en évaluant sur self.X_test via le Trainer.
        On choisit la configuration qui minimise un score composite (ici, basé sur la loss).
        
        Returns
        -------
        best_model : Le modèle LSTMPredictionAnomalyDetector entraîné avec la meilleure configuration.
        best_params : dict des meilleurs hyperparamètres.
        best_score : float, le score composite associé.
        """

        # Définition de la grille d'hyperparamètres
        param_grid = {
            'lstm_hidden_dims': [[64, 64, 32]],
            'linear_hidden_layers': [[100, 50]],
            'linear_activation': [torch.nn.ELU(), torch.nn.ReLU],
            'prediction_horizon': [1], #3, 5, 1
            'learning_rate': [1e-3], #, 5e-4
            'optimizer': ['adam' ] #,'sgd'
        }
        best_score = float('inf')
        best_params = None
        best_detector = None

        # Préparation des DataLoader à partir de self.X_train et self.X_test
        # On transpose pour obtenir la forme (T, N, D) attendue par le modèle
        train_dataset = TensorDataset(self.X_train.transpose(0, 1), self.X_train.transpose(0, 1))
        val_dataset   = TensorDataset(self.X_test.transpose(0, 1), self.X_test.transpose(0, 1))
        train_loader  = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn_lstmp)
        val_loader    = DataLoader(val_dataset, batch_size=32, collate_fn=collate_fn_lstmp)

        # On définit une fonction de validation basée sur la loss
        val_metrics = {
            'loss_0': lambda outputs, targets, inputs, **kwargs: torch.nn.MSELoss()(
                outputs[0] if isinstance(outputs, tuple) else outputs,
                targets[0] if isinstance(targets, tuple) else targets
            )
        }

        
        class SingleInputWrapper(torch.nn.Module):
            def __init__(self, model):
                super(SingleInputWrapper, self).__init__()
                self.model = model
                #self.target_horizon = target_horizon
            def forward(self, inputs):
                # On prend le premier élément du tuple d'entrée
                x = inputs[0]
                #original_horizon = self.model.prediction_horizon #
                # Forcer l'horizon à 1 pour éviter le reshape invalide dans le forward du modèle
                #self.model.prediction_horizon = 1 #
                out = self.model((x,))
                #self.model.prediction_horizon = original_horizon #
                # Transposer pour passer de (T, N, D) à (N, T, D) si nécessaire
                out = out.transpose(0,1)
                #out = out.repeat(1, self.target_horizon, 1) #
                return out
            def grouped_parameters(self):
                # Redirige vers la méthode grouped_parameters du modèle interne
                return self.model.grouped_parameters()


        # Boucle sur les configurations de la grille
        for params in itertools.product(*param_grid.values()):
            param_dict = dict(zip(param_grid.keys(), params))
            start_time = time.time()

            # Instanciation du modèle avec les hyperparamètres en cours
            base_model = LSTMPrediction(
                input_dim=self.input_dim,
                lstm_hidden_dims=param_dict['lstm_hidden_dims'],
                linear_hidden_layers=param_dict['linear_hidden_layers'],
                prediction_horizon=param_dict['prediction_horizon']
            )
            # Encapsulation dans le détecteur d'anomalies
            detector = LSTMPredictionAnomalyDetector(base_model)

            # Choix de l'optimiseur
            if param_dict['optimizer'] == 'adam':
                optimizer = torch.optim.Adam(detector.model.parameters(), lr=param_dict['learning_rate'])
            else:
                optimizer = torch.optim.SGD(detector.model.parameters(), lr=param_dict['learning_rate'])

            # Création d'un Trainer local pour cette configuration
            local_trainer = Trainer(
                train_iter=train_loader,
                val_iter=val_loader,
                optimizer=lambda params: optimizer,
                scheduler=lambda opt: torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[3, 4]),
            )
            # Ajout d'un hook d'early stopping sur la loss (optionnel)
            early_stopping = EarlyStoppingHook(metric='loss_0', invert_metric=False, patience=3)
            local_trainer.add_hook(early_stopping, 'post_validation')

            if not hasattr(detector.model, 'grouped_parameters'):
                detector.model.grouped_parameters = lambda: [detector.model.parameters()]
                
            wrapped_loss = LogCoshLoss() #TorchLossWrapper(torch.nn.MSELoss())
            wrapped_model = SingleInputWrapper(detector.model)
            # Entraînement du modèle avec torch.nn.MSELoss()
            local_trainer.train(wrapped_model, losses=wrapped_loss, num_epochs=5, val_metrics=val_metrics, log_fn=default_log_fn())
            # Évaluation sur le set de validation
            val_result = local_trainer.validate_model_once(wrapped_model, val_metrics, epoch=5, num_epochs=5)
            avg_loss = val_result['loss_0']

            # Ici, le score composite est défini comme la loss moyenne (on peut l'étendre en combinant d'autres métriques)
            composite_score = avg_loss

            elapsed_time = time.time() - start_time
            print(f"Config {param_dict} : Loss = {avg_loss:.4f}, Score = {composite_score:.4f} (Temps: {elapsed_time:.2f}s)")

            if composite_score < best_score:
                best_score = composite_score
                best_params = param_dict
                best_detector = detector

        print("\nMeilleurs hyperparamètres:", best_params, "avec score", best_score)
        self.best_model = best_detector
        self.best_params = best_params
        self.best_score = best_score
        return best_detector, best_params, best_score

    def test_model(self):
        """
        Teste le modèle optimisé en calculant la loss sur le jeu de test.
        Pour LSTMPrediction, la cible est la séquence entière.
        """
        if not hasattr(self, 'best_model'):
            raise ValueError("Veuillez d'abord optimiser les hyperparamètres.")
        
        detector = self.best_model
        criterion = LogCoshLoss() #torch.nn.MSELoss()
        test_dataset = TensorDataset(self.X_test.transpose(0, 1), self.X_test.transpose(0, 1))
        test_loader = DataLoader(test_dataset, batch_size=32, collate_fn=collate_fn_lstmp)

        detector.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch, target in test_loader:
                x = batch
                output = full_sequence_forward(detector.model, x)
                total_loss += criterion(output, x[-detector.model.prediction_horizon:,:,:]).item()
        test_loss = total_loss / len(test_loader)
        print(f"Loss sur l'ensemble de test : {test_loss}")
        self.test_loss = test_loss
        return test_loss
        
    def test_model_with_metrics(self, threshold_percentile):
        """
        Teste le modèle optimisé en calculant non seulement la loss sur le jeu de test,
        mais également plusieurs métriques d'évaluation spécifiques à la détection d'anomalies.
        Ces métriques incluent F1, AUC, AUPRC, TRec et TPrec.
        
        On suppose que la méthode get_labels_and_scores du détecteur retourne
        un tuple (labels, scores) où labels et scores sont des tenseurs de même taille.
        
        Retourne :
            dict: un dictionnaire récapitulant les métriques calculées.
        """

        # On suppose que compute_TRec et compute_TPrec sont disponibles et importées
        # from time_sead_metrics import compute_TRec, compute_TPrec

        if not hasattr(self, 'best_model'):
            raise ValueError("Veuillez d'abord optimiser les hyperparamètres.")

        detector = self.best_model

        # Préparation du DataLoader pour le test (utilisation directe de self.X_test)
        test_dataset = TensorDataset(self.X_test.transpose(0, 1), self.X_test.transpose(0, 1))
        test_loader = DataLoader(test_dataset, batch_size=32, collate_fn=collate_fn_lstmp_test)
        
        class SingleInputWrapper(torch.nn.Module):
            def __init__(self, model):
                super(SingleInputWrapper, self).__init__()
                self.model = model
                #self.target_horizon = target_horizon
            def forward(self, inputs):
                # On prend le premier élément du tuple d'entrée
                x = inputs[0]
                #original_horizon = self.model.prediction_horizon #
                # Forcer l'horizon à 1 pour éviter le reshape invalide dans le forward du modèle
                #self.model.prediction_horizon = 1 #
                out = self.model((x,))
                #self.model.prediction_horizon = original_horizon #
                # Transposer pour passer de (T, N, D) à (N, T, D) si nécessaire
                out = out.transpose(0,1)
                #out = out.repeat(1, self.target_horizon, 1) #
                return out
            def grouped_parameters(self):
                # Redirige vers la méthode grouped_parameters du modèle interne
                return self.model.grouped_parameters()
                
        # Avant de tester, on enveloppe le modèle pour qu'il ne prenne que le premier élément de l'entrée
        detector.model = SingleInputWrapper(detector.model)


        # Récupération des labels et scores d'anomalie via la méthode du détecteur
        labels, scores = detector.get_labels_and_scores(test_loader)
        y_true = labels.numpy()
        anomaly_scores = scores.numpy()

        # Définition d'un seuil pour binariser les scores, par exemple le 95e percentile
        threshold = np.percentile(anomaly_scores, threshold_percentile)
        y_pred = (anomaly_scores > threshold).astype(int)

        # Calcul des métriques classiques
        f1 = f1_score(y_true, y_pred)
        auc = roc_auc_score(y_true, anomaly_scores)
        auprc = average_precision_score(y_true, anomaly_scores)

        # Calcul des métriques spécifiques TimeSeAD (TRec, TPrec)
        try:
            from time_sead_metrics import compute_TRec, compute_TPrec
            t_rec = compute_TRec(y_true, anomaly_scores, threshold)
            t_prec = compute_TPrec(y_true, anomaly_scores, threshold)
        except ImportError:
            t_rec = None
            t_prec = None

        print("Résultats sur le jeu de test :")
        print(f"F1 Score: {f1:.4f}")
        print(f"AUC: {auc:.4f}")
        print(f"AUPRC: {auprc:.4f}")
        if t_rec is not None and t_prec is not None:
            print(f"TRec: {t_rec:.4f}")
            print(f"TPrec: {t_prec:.4f}")
        else:
            print("TRec et TPrec n'ont pas pu être calculés (fonctions non disponibles).")

        # Retourne un dictionnaire récapitulatif
        return {
            "F1": f1,
            "AUC": auc,
            "AUPRC": auprc,
            "TRec": t_rec,
            "TPrec": t_prec,
            "Threshold": threshold
        }


    def plot_anomaly_scores(self):
        """
        Pour chaque fenêtre de X_test, calcule le score d'anomalie en comparant le dernier pas
        de la prédiction (du horizon) à celui de l'entrée, puis ajuste l'erreur avec la moyenne et la
        covariance calculées lors de l'entraînement.
        """
        if not hasattr(self, 'best_model'):
            raise ValueError("Veuillez d'abord optimiser les hyperparamètres.")

        detector = self.best_model
        anomaly_scores = []
        # Pour chaque échantillon (de dimension N) dans X_test
        for i in range(len(self.X_test)):
            # Préparer la fenêtre sous la forme (T, 1, D)
            window = self.X_test[i].unsqueeze(1)  # (T, 1, D)
            with torch.no_grad():
                # Prédiction : (horizon, 1, D)
                pred = detector.model((window,))
            # On utilise le dernier pas de la prédiction
            pred_last = pred[-1, 0, :]     # (D,)
            target_last = window[-1, 0, :]   # (D,)
            error = torch.abs(target_last - pred_last)
            # Ajustement par la moyenne et le produit bilinéaire
            error = error - detector.mean
            score = F.bilinear(error.unsqueeze(0), error.unsqueeze(0), detector.precision.unsqueeze(0))
            anomaly_scores.append(score.squeeze().cpu().numpy())
        anomaly_scores = np.array(anomaly_scores).squeeze()

        plt.figure(figsize=(12, 6))
        plt.plot(anomaly_scores, label="Scores d'anomalie (LSTMPrediction)", color="red", alpha=0.7)
        plt.title("Détection d'anomalies avec LSTMPrediction (Malhotra 2015)")
        plt.xlabel("Index d'échantillon")
        plt.ylabel("Score d'anomalie")
        plt.legend()
        plt.show()

        df_scores = pd.DataFrame({
            "anomaly_score": anomaly_scores,
            "index": np.arange(len(anomaly_scores))
        })

        mean_score = np.mean(anomaly_scores)
        std_score = np.std(anomaly_scores)
        z_scores = (anomaly_scores - mean_score) / std_score

        # Fixation d'un seuil global (par exemple, 3 en Z-score)
        threshold = 3.0
        anomalies = z_scores > threshold
        num_anomalies = np.sum(anomalies)
        ratio_anomalies = num_anomalies / len(z_scores) * 100.0

        print(f"Seuil (Z-score): {threshold:.4f}")
        print(f"Nombre d'anomalies détectées : {num_anomalies}")
        print(f"Ratio d'anomalies : {ratio_anomalies:.2f}%")

        plt.figure(figsize=(10, 6))
        sns.histplot(df_scores["anomaly_score"], bins=50, kde=True)
        plt.title("Distribution des scores d'anomalie")
        plt.xlabel("Score d'anomalie")
        plt.ylabel("Fréquence")
        plt.show()

        self.anomaly_scores = anomaly_scores
        return anomaly_scores


#FONCTIONS LSTM-AE MIRZA
def collate_fn_lstm(batch):
    """ Renvoie (xs, ys) où xs et ys sont de forme (T, B, D) """
    xs, ys = zip(*batch)
    xs = torch.stack(xs).transpose(0, 1)  # (T, B, D)
    ys = torch.stack(ys).transpose(0, 1)
    return (xs,), (ys,)

def full_sequence_forward(model, x):
    """
    Passe avant adaptée pour le LSTM-AE.
    Si le modèle possède l'attribut 'model', on en extrait le tenseur (cas LSTM-AE).
    """
    # Pour LSTM-AE, x doit être de forme (T, B, D)
    if isinstance(x, tuple):
        x = x[0]
    if hasattr(model, 'model'):
        if model.training:
            hidden = model.model.encode(x)
            seq_len = x.shape[0]
            out = model.model.decoder(hidden, seq_len, x)
        else:
            with torch.no_grad():
                hidden = model.model.encode(x)
                seq_len = x.shape[0]
                out = model.model.decoder(hidden, seq_len, x)
        return out
    else:
        if not isinstance(x, tuple):
            x = (x,)
        if model.training:
            return model(x)
        else:
            with torch.no_grad():
                return model(x)

#CLASS TEST LSTM-AE MIRZA
class LSTMAETester:
    """
    Cette classe permet de :
      1. Optimiser les hyperparamètres d'un modèle LSTM-AE via un simple split train/test.
      2. Tester le modèle avec le jeu de test et afficher la loss.
      3. Visualiser le score d'anomalie (calculé sur chaque fenêtre de X_test).
    
    On suppose que :
      - Le modèle LSTM-AE est construit via LSTMAEAnomalyDetector qui encapsule un modèle interne (LSTMAEMirza2018).
      - Le format des tenseurs d'entraînement et de test est (N, T, D).
    """
    
    def __init__(self, X_train, X_test):
        """
        Parameters
        ----------
        X_train : torch.Tensor
            Tenseur d'entraînement de forme (N, T, D)
        X_test : torch.Tensor
            Tenseur de test de forme (N, T, D)
        """
        self.X_train = X_train
        self.X_test = X_test
        self.input_dim = X_train.shape[-1]

    def optimize_hyperparameters(self):
        """
        Recherche sur une grille d'hyperparamètres (hidden_dimensions et latent_pooling)
        et choisit la configuration qui minimise la loss sur le jeu de test.
        
        Retourne
        -------
        best_model : Le modèle LSTM-AE entraîné avec la meilleure configuration.
        best_params : dict des meilleurs hyperparamètres.
        best_loss : float, la loss correspondante.
        """
        import itertools, time
        param_grid = {
            'hidden_dimensions': [[32], [64], [128]],
            'latent_pooling': ['last', 'mean']
        }
        best_loss = float('inf')
        best_params = None
        best_model = None
        collate_fn = collate_fn_lstm  # fonction de collate définie en dehors de la classe
    
        for params in itertools.product(*param_grid.values()):
            param_dict = dict(zip(param_grid.keys(), params))
            start_time = time.time()
    
            # Instanciation du modèle avec la configuration courante
            model = LSTMAEAnomalyDetector(
                LSTMAEMirza2018(input_dimension=self.input_dim, **param_dict)
            )
            
            # Création des DataLoaders pour l'entraînement et le test
            train_loader = DataLoader(
                TensorDataset(self.X_train, self.X_train),
                batch_size=32, shuffle=True, collate_fn=collate_fn
            )
            test_loader = DataLoader(
                TensorDataset(self.X_test, self.X_test),
                batch_size=32, collate_fn=collate_fn
            )
            
            criterion = TorchLossWrapper(torch.nn.MSELoss())
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            
            # Entraînement (5 époques pour accélérer)
            model.train()
            for epoch in range(5):
                for batch, _ in train_loader:
                    x = batch[0] if isinstance(batch, tuple) else batch
                    optimizer.zero_grad()
                    output = full_sequence_forward(model, x)
                    loss = criterion(output, x)
                    loss.backward()
                    optimizer.step()
            
            # Calcul des statistiques (mean et covariance) sur l'erreur du dernier pas de temps
            model.eval()
            all_errors = []
            with torch.no_grad():
                for batch, _ in train_loader:
                    x = batch[0] if isinstance(batch, tuple) else batch
                    output = full_sequence_forward(model, x)
                    # On calcule l'erreur sur le dernier pas de temps uniquement
                    error = torch.abs(x[-1] - output[-1])  # forme (B, D)
                    all_errors.append(error)
            all_errors = torch.cat(all_errors, dim=0)  # forme (N, D) avec N = total des échantillons
            # Calcul de la moyenne des erreurs sur le dernier pas de temps (vecteur de forme (D,))
            model.mean = all_errors.mean(dim=0)
            # Centrer les erreurs et calculer la matrice de covariance
            errors_centered = all_errors - model.mean.unsqueeze(0)
            cov = torch.matmul(errors_centered.T, errors_centered) / (errors_centered.shape[0] - 1)
            # Pour plus de stabilité, ajouter une petite valeur sur la diagonale
            cov.diagonal().add_(1e-5)
            model.precision = torch.inverse(cov)
            
            # Évaluation sur le jeu de test
            total_loss = 0
            with torch.no_grad():
                for batch, _ in test_loader:
                    x = batch[0] if isinstance(batch, tuple) else batch
                    output = full_sequence_forward(model, x)
                    total_loss += criterion(output, x).item()
            test_loss = total_loss / len(test_loader)
            elapsed_time = time.time() - start_time
            print(f"Test Loss avec {param_dict} : {test_loss:.4f} (Temps: {elapsed_time:.2f} sec)")
            
            if test_loss < best_loss:
                best_loss = test_loss
                best_params = param_dict
                best_model = model

        print("\n")
        print(f"Meilleurs hyperparamètres: {best_params} avec Loss {best_loss:.4f}")
        self.best_model = best_model
        self.best_params = best_params
        self.best_loss = best_loss
        return best_model, best_params, best_loss


    def test_model(self):
        """
        Teste le modèle optimisé en calculant la loss sur le jeu de test.
        Pour LSTM-AE, on transpose X_test pour obtenir la forme (T, B, D) attendue par full_sequence_forward.
        """
        if not hasattr(self, 'best_model'):
            raise ValueError("Veuillez d'abord optimiser les hyperparamètres.")
        
        model = self.best_model
        criterion = TorchLossWrapper(torch.nn.MSELoss())
        # Pour LSTM-AE, le modèle attend des tenseurs de forme (T, B, D)
        test_data_transposed = self.X_test.transpose(0, 1)  # (T, B, D)
        with torch.no_grad():
            test_outputs = full_sequence_forward(model, test_data_transposed)
        print("Test outputs shape:", test_outputs.shape)
        last_test_outputs = test_outputs[-1]
        last_targets = test_data_transposed[-1]
        test_loss = criterion(last_test_outputs, last_targets).item()
        print(f"Loss sur l'ensemble de test : {test_loss}")
        self.test_loss = test_loss
        return test_loss

    def plot_anomaly_scores(self):
        """
        Calcule et affiche le score d'anomalie pour chaque fenêtre de X_test.
        Pour chaque fenêtre (indice sur la dimension temporelle), on prépare un tenseur de forme (T, 1, D)
        et on calcule le score d'anomalie avec la méthode compute_online_anomaly_score.
        """
        if not hasattr(self, 'best_model'):
            raise ValueError("Veuillez d'abord optimiser les hyperparamètres.")
        
        model = self.best_model
        anomaly_scores = []
        # Itération sur chaque fenêtre de X_test
        for i in range(len(self.X_test)):
            # Préparer la fenêtre sous la forme (T, 1, D)
            window = self.X_test[i].unsqueeze(0).transpose(0, 1)
            score = model.compute_online_anomaly_score((window,)).detach().cpu().numpy()
            anomaly_scores.append(score)
        anomaly_scores = np.array(anomaly_scores)
        plt.figure(figsize=(12, 6))
        plt.plot(anomaly_scores, label="Scores d'anomalie (LSTM AE)", color="red", alpha=0.7)
        plt.title("Détection d'anomalies avec LSTM-AE")
        plt.xlabel("Index temporel")
        plt.ylabel("Score d'anomalie")
        plt.legend()
        plt.show()
        self.anomaly_scores = anomaly_scores
        return anomaly_scores
        
#FONCTIONS MALHOTRA 2016
def collate_fn_lstm(batch):
     xs, ys = zip(*batch)
     xs = torch.stack(xs)  # (N, T, D)
     ys = torch.stack(ys)  # (N, T, D)
     # Transposition éventuelle si le modèle attend (T, B, D)
     xs = xs.transpose(0, 1)
     ys = ys.transpose(0, 1)
     return xs, ys
def full_sequence_forward(model, x):
     # Si le modèle attend un tuple d'inputs (ex. (x,)), on l'envoie ainsi.
     return model((x,))
     
#CLASS TEST MALHOTRA 2016
class LSTMAEMALHOTRATester:
    """
    Cette classe permet de :
      1. Optimiser les hyperparamètres d'un modèle LSTM-AE basé sur l'approche Malhotra 2016,
         en utilisant un split train/test.
      2. Entraîner le modèle et calculer la loss sur le jeu de test.
      3. Calculer et visualiser le score d'anomalie pour chaque fenêtre de X_test,
         en évaluant l'erreur de reconstruction sur l'ensemble de la séquence (moyenne sur T).

    On suppose que :
      - Le modèle est construit via LSTMAEAnomalyDetector encapsulant un LSTMAEMalhotra2016.
      - Les tenseurs d'entraînement et de test sont de forme (N, T, D).
    """
    def __init__(self, X_train, X_test):
        """
        Parameters
        ----------
        X_train : torch.Tensor
            Tenseur d'entraînement de forme (N, T, D)
        X_test : torch.Tensor
            Tenseur de test de forme (N, T, D)
        """
        self.X_train = X_train
        self.X_test = X_test
        self.input_dim = X_train.shape[-1]

    def optimize_hyperparameters(self):
        """
        Recherche sur une grille d'hyperparamètres (ici, hidden_dimensions et latent_pooling)
        et choisit la configuration qui minimise la loss sur le jeu de test.
        Pour Malhotra, on fixe le pooling à 'last'.

        Returns
        -------
        best_model : Le modèle LSTM-AE entraîné avec la meilleure configuration.
        best_params : dict des meilleurs hyperparamètres.
        best_loss : float, la loss correspondante.
        """
        param_grid = {
            'hidden_dimensions': [[32]] # ,[64], [128]
        }
        best_loss = float('inf')
        best_params = None
        best_model = None
        collate_fn = collate_fn_lstm  # fonction de collate définie ailleurs

        for params in itertools.product(*param_grid.values()):
            param_dict = dict(zip(param_grid.keys(), params))
            start_time = time.time()

            # Instanciation du modèle via LSTMAEMalhotra2016
            model = LSTMAEAnomalyDetector(
                LSTMAEMalhotra2016(input_dimension=self.input_dim, **param_dict)
            )
            
            # Création des DataLoaders pour l'entraînement et le test
            train_loader = DataLoader(
                TensorDataset(self.X_train, self.X_train),
                batch_size=32, shuffle=True, collate_fn=collate_fn
            )
            test_loader = DataLoader(
                TensorDataset(self.X_test, self.X_test),
                batch_size=32, collate_fn=collate_fn
            )
            
            criterion = TorchLossWrapper(torch.nn.MSELoss())
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            
            # Entraînement (par exemple, 5 époques)
            model.train()
            for epoch in range(5):
                for batch, _ in train_loader:
                    # Selon votre collate_fn, x est de forme (T, B, D)
                    x = batch[0] if isinstance(batch, tuple) else batch
                    optimizer.zero_grad()
                    output = full_sequence_forward_lstm(model.model, x)
                    loss = criterion(output, x)
                    loss.backward()
                    optimizer.step()
            
            # Calcul des statistiques sur la reconstruction
            # Ici, on calcule l'erreur sur toute la séquence (moyenne sur la dimension temporelle)
            model.eval()
            all_errors = []
            sum_error = 0
            total_samples = 0
            with torch.no_grad():
                for batch, _ in train_loader:
                    x = batch[0] if isinstance(batch, tuple) else batch  # (T, B, D)
                    output = full_sequence_forward_lstm(model.model, x)
                    # Calcul de l'erreur absolue sur toute la séquence, moyennée sur T
                    error = torch.abs(x - output).mean(dim=0)  # (B, D)
                    all_errors.append(error)
                    sum_error += torch.sum(error, dim=0)
                    total_samples += error.shape[0]
            all_errors = torch.cat(all_errors, dim=0)  # (N_total, D)
            mean_error = sum_error / total_samples  # (D,)
            
            # Centrer les erreurs et calculer la covariance sur l'ensemble des erreurs
            errors_centered = all_errors - mean_error.unsqueeze(0)
            cov = torch.matmul(errors_centered.T, errors_centered) / (errors_centered.shape[0] - 1)
            cov.diagonal().add_(1e-5)
            model.mean = mean_error
            model.precision = torch.inverse(cov)
            
            # Évaluation sur le jeu de test (calcul de la loss sur la séquence entière)
            total_loss = 0
            with torch.no_grad():
                for batch, _ in test_loader:
                    x = batch[0] if isinstance(batch, tuple) else batch
                    output = full_sequence_forward_lstm(model.model, x)
                    total_loss += criterion(output, x).item()
            test_loss = total_loss / len(test_loader)
            elapsed_time = time.time() - start_time
            print(f"Test Loss avec {param_dict} : {test_loss:.4f} (Temps: {elapsed_time:.2f} sec)")
            
            if test_loss < best_loss:
                best_loss = test_loss
                best_params = param_dict
                best_model = model

        print("\nMeilleurs hyperparamètres:", best_params, "avec Loss", best_loss)
        self.best_model = best_model
        self.best_params = best_params
        self.best_loss = best_loss
        return best_model, best_params, best_loss

    def test_model(self):
        """
        Teste le modèle optimisé en calculant la loss sur le jeu de test.
        Pour LSTM-AE, le modèle attend des tenseurs de forme (T, B, D).
        """
        if not hasattr(self, 'best_model'):
            raise ValueError("Veuillez d'abord optimiser les hyperparamètres.")
        
        model = self.best_model
        criterion = TorchLossWrapper(torch.nn.MSELoss())
        # Transposition pour obtenir (T, B, D)
        test_data_transposed = self.X_test.transpose(0, 1)
        with torch.no_grad():
            test_outputs = full_sequence_forward(model.model, test_data_transposed)
        print("Test outputs shape:", test_outputs.shape)
        test_loss = criterion(test_outputs, test_data_transposed).item()
        print(f"Loss sur l'ensemble de test : {test_loss}")
        self.test_loss = test_loss
        return test_loss

    def compute_online_anomaly_score(self, window):
        """
        Calcule le score d'anomalie pour une fenêtre donnée en utilisant l'erreur
        de reconstruction sur toute la séquence (moyennée sur la dimension temporelle).

        Parameters
        ----------
        window : torch.Tensor
            Tenseur de forme (T, 1, D) représentant une fenêtre.
        
        Returns
        -------
        score : torch.Tensor
            Score d'anomalie pour la fenêtre (1 ou B, selon la taille du batch).
        """
        model = self.best_model
        model.eval()
        with torch.no_grad():
            # Ici, on utilise le modèle sous-jacent pour obtenir la reconstruction
            output = full_sequence_forward(model.model, window)
        # Calcul de l'erreur sur toute la séquence et moyennage sur T
        error = torch.abs(window - output).mean(dim=0)  # (1, D)
        error = error - model.mean  # Centrer par rapport à la moyenne calculée
        result = F.bilinear(error, error, model.precision.unsqueeze(0))
        return result.squeeze(-1)

    def plot_anomaly_scores(self):
        """
        Pour chaque fenêtre de X_test, calcule le score d'anomalie (en ligne)
        et affiche la courbe des scores.
        """
        if not hasattr(self, 'best_model'):
            raise ValueError("Veuillez d'abord optimiser les hyperparamètres.")
        
        model = self.best_model
        anomaly_scores = []
        # Itération sur chaque fenêtre de X_test (chaque échantillon de la dimension N)
        for i in range(len(self.X_test)):
            # Préparer la fenêtre sous la forme (T, 1, D)
            window = self.X_test[i].unsqueeze(0).transpose(0, 1)
            score = self.compute_online_anomaly_score(window).detach().cpu().numpy()
            anomaly_scores.append(score)
        anomaly_scores = np.array(anomaly_scores)

        anomaly_scores = np.array(anomaly_scores).squeeze()
        
        plt.figure(figsize=(12, 6))
        plt.plot(anomaly_scores, label="Scores d'anomalie (LSTM-AE Malhotra)", color="blue", alpha=0.7)
        plt.title("Détection d'anomalies avec LSTM-AE (Malhotra 2016)")
        plt.xlabel("Index temporel")
        plt.ylabel("Score d'anomalie")
        plt.legend()
        plt.show()

        df_scores = pd.DataFrame({
        "anomaly_score": anomaly_scores,
        "index": np.arange(len(anomaly_scores))
        })

        mean_score = np.mean(anomaly_scores)
        std_score = np.std(anomaly_scores)
        z_scores = (anomaly_scores - mean_score) / std_score
    
        # On fixe un seuil global sur les Z-scores (par exemple, 3)
        threshold = 3.0
        anomalies = z_scores > threshold
        num_anomalies = np.sum(anomalies)
        ratio_anomalies = num_anomalies / len(z_scores) * 100.0
    
        print(f"Seuil (Z-score): {threshold:.4f}")
        print(f"Nombre d'anomalies détectées : {num_anomalies}")
        print(f"Ratio d'anomalies : {ratio_anomalies:.2f}%")

        plt.figure(figsize=(10, 6))
        sns.histplot(df_scores["anomaly_score"], bins=50, kde=True)
        plt.title("Distribution des scores d'anomalie")
        plt.xlabel("Score d'anomalie")
        plt.ylabel("Fréquence")
        plt.show()
        self.anomaly_scores = anomaly_scores
        return anomaly_scores
        self.anomaly_scores = anomaly_scores
        return anomaly_scores
        
#TEST TCN-AE SANS CLASSE
'''
print("X_test_tensor shape:", X_test_tensor.shape)
print("X_test_tensor transposé shape:", X_test_tensor.transpose(0, 1).shape)

# Récupérer la dimension d'entrée
input_dim = X_train_tensor.shape[-1]  

# Instancier le modèle TCN-AE et le détecteur associé
model = TCNAEAnomalyDetector(TCNAE(input_dimension=input_dim))

# Définir une fonction de collate personnalisée (sans transposition)
def collate_fn(batch):
    xs, ys = zip(*batch)
    xs = torch.stack(xs)  # (B, T, D)
    ys = torch.stack(ys)  # (B, T, D)
    return (xs,), (ys,)

# Créer le DataLoader avec la fonction collate
train_loader = DataLoader(
    TensorDataset(X_train_tensor, X_train_tensor),
    batch_size=32,
    shuffle=True,
    collate_fn=collate_fn
)

# Vérifier un batch d'exemple
sample_batch = next(iter(train_loader))
inputs, targets = sample_batch
print("Inputs type:", type(inputs))         # Devrait être un tuple
print("Inputs shape:", inputs[0].shape)      # Attendu : (B, T, D)
print("Targets shape:", targets[0].shape)     # Attendu : (B, T, D)

# Entraîner le modèle
model.fit(train_loader)
if not hasattr(model, 'mean') or not hasattr(model, 'precision'):
    print("Erreur : Les attributs 'mean' et 'precision' ne sont pas définis.")
else:
    print("Les attributs 'mean' et 'precision' sont correctement initialisés.")

# Fonction pour obtenir la reconstruction complète via teacher forcing
def full_sequence_forward(model, x):
    with torch.no_grad():
        out = model.model((x,))
    return out

# Test sur un tenseur synthétique (attention : pour TCN-AE, la forme attendue est (B, T, D))
B, T, D = 32, 50, input_dim
x_test = torch.rand(B, T, D)
output = full_sequence_forward(model, x_test)
print("Sortie pour tenseur synthétique :", output.shape)
if output.dim() > 0:
    last_output = output[:, -1, :]  # Extraire le dernier pas pour chaque exemple du batch
    print("Dernier pas de sortie :", last_output.shape)

# Fonction de test sur l'ensemble de test
def test_model(model, test_data):
    criterion = nn.MSELoss()
    with torch.no_grad():
        test_outputs = full_sequence_forward(model, test_data)
    
    print("test_outputs shape:", test_outputs.shape)
    last_test_outputs = test_outputs[:, -1, :]
    print("last_test_outputs shape:", last_test_outputs.shape)
    last_targets = test_data[:, -1, :]
    print("last_targets shape:", last_targets.shape)
    test_loss = criterion(last_test_outputs, last_targets).item()
    print(f"Loss sur l'ensemble de test : {test_loss}")

test_model(model, X_test_tensor)

#Calcul d'un score d'anomalie pour chaque fenêtre de X_test_tensor
anomaly_scores = []
for i in range(len(X_test_tensor)):
    window = X_test_tensor[i].unsqueeze(0)
    score = model.compute_online_anomaly_score((window,)).detach().cpu().numpy()
    anomaly_scores.append(score)

anomaly_scores = np.array(anomaly_scores)


'''
'''
plt.figure(figsize=(12, 6))
plt.plot(anomaly_scores, label="Scores d'anomalie (TCN-AE)", color="blue", alpha=0.7)
plt.title("Détection d'anomalies avec TCN-AE (score d'anomalie pour chaque fenêtre)")
plt.xlabel("Index temporel")
plt.ylabel("Score d'anomalie")
plt.legend(loc="upper right")
plt.show()
'''

#TEST OMNI ANOMALY (TRES LONG DONC MASQUE)
'''
Pour réduire davantage ce temps d'exécution, voici quelques suggestions complémentaires:

• Réduire encore le nombre d'échantillons Monte Carlo
Si vous passez par 4 échantillons, vous pouvez expérimenter avec 2 ou même 1 échantillon, au risque d'une estimation moins précise.

• Optimisation par vectorisation
Vérifiez si certaines boucles internes (par exemple dans le filtre de Kalman ou dans le traitement de normalizing flow) peuvent être davantage vectorisées pour réduire le surcoût des itérations en Python.

• Utiliser TorchScript plus largement
Vous pouvez essayer de compiler l'ensemble du modèle (pas uniquement detector.model) avec TorchScript pour réduire l'overhead de Python sur CPU.

• Profilage et optimisation spécifique
Il peut être utile de profiler votre code pour identifier précisément les parties les plus coûteuses et optimiser uniquement ces sections.

Ces ajustements devraient vous permettre de réduire davantage le temps d'exécution, mais gardez à l'esprit que les modèles complexes comme OmniAnomaly restent gourmands en ressources sur CPU par rapport à une exécution sur GPU.

'''
'''

# Limiter le nombre de threads CPU pour optimiser les performances
torch.set_num_threads(4)

# Supposons que X_train_tensor et X_test_tensor soient déjà définis
input_dim = X_train_tensor.shape[-1]  # Dernière dimension du tenseur

print("X_test_tensor shape:", X_test_tensor.shape)
print("X_test_tensor transposé shape:", X_test_tensor.transpose(0, 1).shape)

input_dim = X_train_tensor.shape[-1]  # Dernière dimension du tenseur

# Création du modèle OmniAnomaly et du détecteur associé
model_omni = OmniAnomaly(input_dim=input_dim, latent_dim=3)  # latent_dim par défaut à 3
# Réduire num_mc_samples de 128 à 16 pour accélérer le calcul
detector = OmniAnomalyDetector(model_omni, num_mc_samples=2)

# ----- Définir une fonction de collate personnalisée -----
def collate_fn(batch):
    xs, ys = zip(*batch)
    xs = torch.stack(xs).transpose(0, 1)  # (T, B, D)
    ys = torch.stack(ys).transpose(0, 1)  # (T, B, D)
    return (xs,), (ys,)

# ----- Créer le DataLoader -----
train_loader = DataLoader(
    TensorDataset(X_train_tensor, X_train_tensor),
    batch_size=32,
    shuffle=True,
    collate_fn=collate_fn
)

# Affichage d'un échantillon pour vérifier les dimensions
sample_batch = next(iter(train_loader))
inputs, targets = sample_batch
print("Inputs type:", type(inputs))       # Doit être un tuple
print("Inputs shape:", inputs[0].shape)    # Devrait afficher (T, B, D)
print("Targets shape:", targets[0].shape)   # Devrait afficher (T, B, D)

# ----- Boucle d'entraînement pour OmniAnomaly -----
loss_fn = OmniAnomalyLoss()
optimizer = torch.optim.Adam(model_omni.parameters(), lr=1e-3)
num_epochs = 1  # Pour un test rapide

for epoch in range(num_epochs):
    epoch_loss = 0.
    for batch in train_loader:
        inputs, targets = batch
        optimizer.zero_grad()
        # Utiliser un nombre réduit d'échantillons MC pour accélérer l'entraînement
        outputs = model_omni(inputs, num_samples=16)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(train_loader):.4f}")

# ----- Test sur un tenseur synthétique -----
T, B, D = 50, 32, input_dim
x_test = torch.rand(T, B, D)
with torch.no_grad():
    anomaly_score_synth = detector.compute_online_anomaly_score((x_test,))
print("Anomaly score sur tenseur synthétique:", anomaly_score_synth)

# ----- Détection d'anomalies sur X_test_tensor -----
anomaly_scores = []
for i in range(len(X_test_tensor)):
    # Préparer chaque fenêtre avec la forme (T, 1, D)
    window = X_test_tensor[i].unsqueeze(0).transpose(0, 1)
    with torch.no_grad():
        score = detector.compute_online_anomaly_score((window,)).detach().cpu().numpy()
    anomaly_scores.append(score)
anomaly_scores = np.array(anomaly_scores)

plt.figure(figsize=(12, 6))
plt.plot(anomaly_scores, label="Scores d'anomalie (OmniAnomaly)", color="blue", alpha=0.7)
plt.title("Détection d'anomalies avec OmniAnomaly")
plt.xlabel("Index temporel")
plt.ylabel("Score d'anomalie")
plt.legend(loc="upper right")
plt.show()
'''
