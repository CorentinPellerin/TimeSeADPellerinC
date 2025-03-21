import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import importlib
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer, PowerTransformer
from timesead.data.dataset import BaseTSDataset
from timesead.data.transforms.dataset_source import DatasetSource
from timesead.data.transforms.pipeline_dataset import PipelineDataset 
from timesead.data.transforms.window_transform import WindowTransform
from timesead.data.transforms.target_transforms import PredictionTargetTransform

    
#Pour voir quelles sont les colonnes vides dans tous les fichiers In Ordnung
folder_path = "C:/Users/pellerinc/TimeSeAD-master/Daten_von_APL/Export/i.O"
file_paths = glob.glob(os.path.join(folder_path, "WLTC_*.csv"))  # Trouver tous les fichiers WLTC_i.csv

empty_columns_per_file = {}

for file_path in file_paths:
    try:
        df = pd.read_csv(file_path, delimiter=';', skiprows=1, low_memory=False)
        empty_columns = df.columns[df.isna().all()]  # Colonnes totalement vides
        empty_columns_per_file[os.path.basename(file_path)] = list(empty_columns)
    except Exception as e:
        empty_columns_per_file[os.path.basename(file_path)] = f"Erreur lors de la lecture : {e}"

df_empty_columns = pd.DataFrame(list(empty_columns_per_file.items()), columns=["Fichier", "Colonnes Vides"])

# print(df_empty_columns)

#df.head, .tail, .info, .describe, .shape

def UpstreamDataVisualization(df2):
    fig, axes = plt.subplots(nrows=7, ncols=7, figsize=(20, 15))
    axes = axes.flatten()

    for i, col in enumerate(df2.columns):
        df2[col].plot(ax=axes[i], title=col, alpha=0.7)

    plt.tight_layout()
    plt.show()
    
    groups = {
    "Moteur": ["rpm", "Nm", "%"],
    "Températures": ["°C.1", "°C.2", "°C.3"],
    "Pressions": ["hPa"],
    "Émissions": ["ppm", "g/s", "mg/s"],
}

    for category, cols in groups.items():
        plt.figure(figsize=(10, 5))
        sns.boxplot(data=df2[cols])
        plt.xticks(rotation=45)
        plt.title(f"Distribution des variables - {category}")
        plt.show()
    
    #AFFICHE LA DISTRIBUTION DES VARIABLES
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df2)
    plt.xticks(rotation=90)
    plt.title("Distribution des variables AVANT correction")
    plt.show()
    #Visualisation de l'ensemble des features


'''
feature = "ppm" if "ppm" in df2.columns else df2.columns[0]

# Introduire artificiellement des valeurs manquantes pour simuler une interpolation
df_test = df2[[feature]].copy()
df_test.iloc[::10] = np.nan  # Remplacer 10% des données par NaN

# Différentes méthodes d'interpolation à tester
methods = ["linear", "spline", "cubic", "nearest"]

plt.figure(figsize=(12, 6))
plt.plot(df_test.index, df2[feature], label="Original", alpha=0.5)

for method in methods:
    df_interpolated = df_test.interpolate(method=method, order=3 if method == "spline" else None)
    plt.plot(df_interpolated.index, df_interpolated[feature], label=f"Interpolé ({method})", alpha=0.7)


plt.legend()
plt.title(f"Comparaison des interpolations pour {feature}")
plt.xlabel("Temps")
plt.ylabel("Valeur")
plt.show()
'''

MODEL_MAP_original = {
    "Reconstruction": {
        "lstm_ae": "from timesead.models.reconstruction.lstm_ae import LSTMAEAnomalyDetector",
        "tcn_ae": "from timesead.models.reconstruction.tcn_ae import TCNAEAnomalyDetector",
        "usad": "from timesead.models.reconstruction.usad import USADAnomalyDetector",
        "mscred": "from timesead.models.reconstruction.mscred import MSCREDAnomalyDetector",
        "genad": "from timesead.models.reconstruction.genad import GENADDetector",
        "anom_trans": "from timesead.models.reconstruction.anom_trans import AnomalyTransformer",
        "stgat_mad": "from timesead.models.reconstruction.stgat_mad import STGATMAD",
    },
    "prediction": {
        "gdn": "from timesead.models.prediction.gdn import GDN",
        "lstm_prediction": "from timesead.models.prediction.lstm_prediction import LSTMPredictionAnomalyDetector",
        "tcn_prediction": "from timesead.models.prediction.tcn_prediction import TCNPredictionAnomalyDetector",
    },
    "generative-GAN": {
        "madgan": "from timesead.models.generative.madgan import MADGANAnomalyDetector",
        "tadgan": "from timesead.models.generative.tadgan import TADGANAnomalyDetector",
        "beatgan": "from timesead.models.generative.beatgan import BeatGAN",
        "lstm_vae_gan": "from timesead.models.generative.lstm_vae_gan import LSTM_VAE_GAN",
    },
    "generative-VAE": {
        "lstm_vae": "from timesead.models.generative.lstm_vae import LSTMVAE",
        "gmm_gru_vae": "from timesead.models.generative.gmm_gru_vae import GMMGRUVAE",
        "sis_vae": "from timesead.models.generative.sis_vae import SISVAE",
        "omni_anomaly": "from timesead.models.generative.omni_anomaly import OmniAnomaly",
        "donut": "from timesead.models.generative.donut import DonutAnomalyDetector",
    },
    "other": {
        "thoc": "from timesead.models.other.thoc import THOCAnomalyDetector",
        "mtad_gat": "from timesead.models.other.mtad_gat import MTAD_GATAnomalyDetector",
        "ncad": "from timesead.models.other.ncad import NCADAnomalyDetector",
        "lstm_ae_ocsvm": "from timesead.models.other.lstm_ae_ocsvm import LSTMAE_OCSVM",
    },
}

MODEL_MAP = {
    "prediction": {
        "tcn_prediction": "timesead.models.prediction.tcn_prediction.TCNPredictionAnomalyDetector",
        "gdn": "timesead.models.prediction.gdn.GDN",
        "lstm_prediction": "timesead.models.prediction.lstm_prediction.LSTMPredictionAnomalyDetector"
    },
    "reconstruction": {
        "lstm_ae": "timesead.models.reconstruction.lstm_ae.LSTMAEAnomalyDetector",
        "tcn_ae": "timesead.models.reconstruction.tcn_ae.TCNAEAnomalyDetector",
        "usad": "timesead.models.reconstruction.usad.USADAnomalyDetector",
        "mscred": "timesead.models.reconstruction.mscred.MSCREDAnomalyDetector",
        "genad": "timesead.models.reconstruction.genad.GENADDetector",
        "anom_trans": "timesead.models.reconstruction.anom_trans.AnomalyTransformer",
        "stgat_mad": "timesead.models.reconstruction.stgat_mad.STGATMAD"
    },
    "generative-GAN": {
        "madgan": "timesead.models.generative.madgan.MADGANAnomalyDetector",
        "tadgan": "timesead.models.generative.tadgan.TADGANAnomalyDetector",
        "beatgan": "timesead.models.generative.beatgan.BeatGAN",
        "lstm_vae_gan": "timesead.models.generative.lstm_vae_gan.LSTM_VAE_GAN"
    },
    "generative-VAE": {
        "lstm_vae": "timesead.models.generative.lstm_vae.LSTMVAE",
        "gmm_gru_vae": "timesead.models.generative.gmm_gru_vae.GMMGRUVAE",
        "sis_vae": "timesead.models.generative.sis_vae.SISVAE",
        "omni_anomaly": "timesead.models.generative.omni_anomaly.OmniAnomaly",
        "donut": "timesead.models.generative.donut.DonutAnomalyDetector"
    },
    "other": {
        "thoc": "timesead.models.other.thoc.THOCAnomalyDetector",
        "mtad_gat": "timesead.models.other.mtad_gat.MTAD_GATAnomalyDetector",
        "ncad": "timesead.models.other.ncad.NCADAnomalyDetector",
        "lstm_ae_ocsvm": "timesead.models.other.lstm_ae_ocsvm.LSTMAE_OCSVM"
    }
}
class SingleSequenceDataset(BaseTSDataset):
    def __init__(self, tensor):
        """
        :param tensor: tenseur de forme (T, num_features) représentant une seule série temporelle.
        """
        self.tensor = tensor

    def __len__(self):
        return 1

    def __getitem__(self, index):
        if index != 0:
            raise IndexError("SingleSequenceDataset n'a qu'une seule séquence.")
        # Retourne la série complète en entrée ET en cible
        return (self.tensor,), (self.tensor,)

    @property
    def seq_len(self):
        return self.tensor.shape[0]

    @property
    def num_features(self):
        return self.tensor.shape[1]
        
    @staticmethod
    def get_default_pipeline() -> dict:
        """
        Retourne le pipeline par défaut pour ce dataset.
        Ici, on retourne un dictionnaire vide.
        """
        return {}
        
    @staticmethod
    def get_feature_names() -> list:
        """
        Retourne les noms des features.
        Cette implémentation se base sur le nombre de features enregistré lors de l'initialisation.
        """
        num_features = getattr(SingleSequenceDataset, "_num_features", 0)
        return [f"feature_{i+1}" for i in range(num_features)]
            
class UpstreamDataPreparation:
    def __init__(self, method=None, threshold=None, seq_length=None, scale_method=None, interpolation_method=None, window_params=None):
        self.method = method  # Méthode de détection des outliers (IQR ou MAD)
        self.threshold = threshold  # Seuil pour détecter les outliers
        self.seq_length = seq_length  # Longueur des séquences temporelles
        self.scale_method = scale_method  # Méthode de normalisation
        self.interpolation_method = interpolation_method
        self.scaler = None  # Scaler à réutiliser
        self.window_params = window_params
        
    def data_collection(self, file_path, time_col):
        df = pd.read_csv(file_path, delimiter=';', skiprows=1, low_memory=False) #skiprows a adapter selon la position des colonnes du fichier
        
        # Vérifier si la colonne de dates est présente dans le DataFrame
        if time_col in df.columns:
            df[time_col] = pd.to_numeric(df[time_col])
            df.set_index(time_col, inplace=True)

            empty_columns = df.columns[df.isna().all()]
            print(f"Colonnes totalement vides avant suppression : {list(empty_columns)}")

            df_cleaned = df.dropna(axis=1, how="all")
            print("Colonnes supprimées !")
                
            if not df.index.is_monotonic_increasing: #peu utile car fichiers APL normalement bien indéxés
                print("L'index n'est pas trié. Tri en cours...")
                df.sort_index(inplace=True)
                print("L'index a été trié avec succès.")
            
        else:
            print(f"La colonne '{time_col}' n'a pas été trouvée. L'index par défaut sera utilisé.")
        
    
        return df_cleaned

    def compute_dynamic_thresholds(self, data):
        """
        Calcule un seuil dynamique basé sur la variance de chaque feature.
        """
        variances = data.var()
        dynamic_thresholds = variances * 1.5
        return dynamic_thresholds

    def detect_motor_anomalies(self, data):
        """
        Détecte uniquement ce qui s'apparentent a des anomalies moteur
        """
        anomalies = pd.DataFrame(False, index=data.index, columns=data.columns)

        if self.threshold is None:
            thresholds = self.compute_dynamic_thresholds(data)
        else:
            thresholds = pd.Series(self.threshold, index=data.columns)
        
        if self.method == "IQR": #Interquartile Range
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            anomalies = (data < (Q1 - thresholds * IQR)) | (data > (Q3 + thresholds * IQR))
        elif self.method == "MAD": #Median Absolute Deviation
            median = data.median()
            mad = (data - median).abs().median()
            anomalies = (data - median).abs() > (thresholds * 1.4826 * mad)
        
        return anomalies #renvoie un DataFrame anomalies de même taille que data, où chaque valeur est True (anomalie moteur) ou False.

    def preprocess_data(self, data):
        """
        Prétraitement des données : correction des erreurs capteurs et détection des anomalies moteur
        """
        print("Détection des anomalies moteur en cours...")
        anomalies_motor = self.detect_motor_anomalies(data)

        total_anomalies = anomalies_motor.sum().sum()
        total_values = data.size

        data_clean = data.copy()

        # Supprimer uniquement les erreurs capteurs avant d'entraîner le modèle
        for col in data.columns:
            mask = anomalies_motor[col]
            data_clean.loc[mask, col] = np.nan  # Remplace uniquement les vraies erreurs

        nan_ratio = data_clean.isna().sum().sum() / data_clean.size
        method = self.interpolation_method

        if nan_ratio < 0.02:  # Seuil de 2% pour suppression des NaN
            print(f"Ratio de NaN ({nan_ratio:.2%}) sous le seuil de 2%, suppression des NaN.")
            data_clean = data_clean.dropna()
        else:
            #reconstruire les NaN du tableau (erreurs capteurs initialement présentes + outliers moteurs) par interpolation
            print(f"Ratio de NaN ({nan_ratio:.2%}) supérieur au seuil de 2%, interpolation appliquée.")
            data_clean = data_clean.interpolate(method=method).bfill()

        total_clean_values = data_clean.size        
        anomaly_ratio = (total_anomalies / total_clean_values) * 100

        print(f"Nombre total d'anomalies moteur détectées : {total_anomalies}")
        print(f"Nombre total de valeurs dans le fichier : {total_values}")
        print(f"Ratio anomalies / total valeurs nettoyées : {anomaly_ratio:.2f}%")
        print(f"Nombre total de valeurs après nettoyage : {total_clean_values}")

        return data_clean, anomalies_motor



    def normalize_data(self, train_data, test_data=None):
        """
        Normalisation des données avec MinMaxScaler ou StandardScaler.
        """
        scale_method = self.scale_method.lower()
    
        if scale_method == "minmax": #pas utile
            self.scaler = MinMaxScaler(feature_range=(0, 1))
        elif scale_method == "standard":
            self.scaler = StandardScaler()
        elif scale_method == "robust":
            self.scaler = RobustScaler()
        elif scale_method == "quantile": #pas utile
            # Vous pouvez choisir 'uniform' ou 'normal' pour output_distribution
            self.scaler = QuantileTransformer(output_distribution='uniform', random_state=0)
        elif scale_method == "power":
            # Yeo-Johnson gère les données négatives et positives
            self.scaler = PowerTransformer(method='yeo-johnson')
        else:
            raise ValueError("Méthode de normalisation non reconnue. Utilisez 'minmax', 'standard', 'robust', 'quantile' ou 'power'.")
    
        
        train_data_scaled = pd.DataFrame(self.scaler.fit_transform(train_data), 
                                         columns=train_data.columns, index=train_data.index)
        
        if test_data is not None:
            test_data_scaled = pd.DataFrame(self.scaler.transform(test_data), 
                                            columns=test_data.columns, index=test_data.index)
            return train_data_scaled, test_data_scaled
        
        return train_data_scaled
    
    def create_sequences(self, data):
        """
        Transformation en séquences temporelles.
        """
        data_np = data.to_numpy()
        if len(data_np) < self.seq_length:
            raise ValueError(f"Pas assez de données ({len(data_np)}) pour créer des séquences de longueur {self.seq_length}")
        sequences = np.array([data_np[i:i + self.seq_length] for i in range(len(data_np) - self.seq_length)])
        return torch.tensor(sequences, dtype=torch.float32)
    
    def process(self, data, train_size=0.8, use_pipeline=True):
        """
        Pipeline complet : suppression des erreurs capteurs, détection d'anomalies moteur,
        normalisation et conversion en séquences temporelles via WindowTransform.
        
        Cette méthode remplace l'ancienne fonction create_sequences en utilisant un pipeline :
          1. La série complète est convertie en tenseur.
          2. Elle est encapsulée dans un SingleSequenceDataset.
          3. Un DatasetSource avec axe 'time' est créé.
          4. WindowTransform est appliqué pour extraire des fenêtres glissantes.
        """
        data_clean, anomalies_motor = self.preprocess_data(data)
        print("Normalisation des données...")
        train_data = data_clean.iloc[:int(len(data_clean) * train_size)]
        test_data = data_clean.iloc[int(len(data_clean) * train_size):]
        train_data, test_data = self.normalize_data(train_data, test_data)
        
        if use_pipeline:
            # Conversion des DataFrame en tenseurs représentant la série complète
            train_tensor = torch.tensor(train_data.to_numpy(), dtype=torch.float32)
            test_tensor = torch.tensor(test_data.to_numpy(), dtype=torch.float32)
            
            # Création d'un dataset encapsulant la série complète
            train_dataset = SingleSequenceDataset(train_tensor)
            test_dataset = SingleSequenceDataset(test_tensor)
            
            # Création d'un DatasetSource avec axe 'time'
            train_source = DatasetSource(train_dataset, start=0, end=train_dataset.seq_len, axis='time')
            test_source = DatasetSource(test_dataset, start=0, end=test_dataset.seq_len, axis='time')
            
            # Application de WindowTransform pour découper en fenêtres glissantes
            if self.window_params is not None:
                window_size = self.window_params.get("window_size")
                step_size = self.window_params.get("step_size", 1)
                reverse = self.window_params.get("reverse", False)
                train_source = WindowTransform(train_source, window_size=window_size, step_size=step_size, reverse=reverse)
                test_source = WindowTransform(test_source, window_size=window_size, step_size=step_size, reverse=reverse)
                
                if self.window_params.get("prediction_horizon") is not None:
                    horizon = self.window_params["prediction_horizon"]
                    train_source = PredictionTargetTransform(train_source, window_size=window_size, prediction_horizon=horizon, replace_labels=True)
                    test_source = PredictionTargetTransform(test_source, window_size=window_size, prediction_horizon=horizon, replace_labels=False)

            
            # Création du pipeline final
            train_pipeline = PipelineDataset(train_source)
            test_pipeline = PipelineDataset(test_source)
            
            return train_pipeline, test_pipeline, anomalies_motor
        else:
            # Si l'on ne souhaite pas utiliser le pipeline, on retourne directement les tenseurs de la série complète
            train_tensor = torch.tensor(train_data.to_numpy(), dtype=torch.float32)
            test_tensor = torch.tensor(test_data.to_numpy(), dtype=torch.float32)
            return train_tensor, test_tensor, anomalies_motor
        
    def process_with_DSsource(self, data, train_size=0.8):
        """
        Pipeline complet : nettoyage, normalisation et création de sources de dataset.
        """
        
        data_clean, anomalies_motor = self.preprocess_data(data)
        
        print("Normalisation des données...")
        train_data = data_clean.iloc[:int(len(data_clean) * train_size)]
        test_data = data_clean.iloc[int(len(data_clean) * train_size):]
        train_data, test_data = self.normalize_data(train_data, test_data)
        
        # Transformation des DataFrames en objets BaseTSDataset
        train_dataset = BaseTSDataset(train_data, seq_len=self.seq_length)
        test_dataset = BaseTSDataset(test_data, seq_len=self.seq_length)
        
        # Création de DatasetSource pour bénéficier d'un découpage flexible (exemple ici en mode 'batch')
        train_source = DatasetSource(train_dataset, start=0, end=len(train_dataset), axis='batch')
        test_source = DatasetSource(test_dataset, start=0, end=len(test_dataset), axis='batch')
        
        # Ces DatasetSource peuvent ensuite être utilisés dans le pipeline d'entraînement
        torch.save(train_source, "train_source.pt")
        torch.save(test_source, "test_source.pt")
        print("Les DatasetSource ont été sauvegardés avec succès.")
        
        return train_source, test_source, anomalies_motor
        