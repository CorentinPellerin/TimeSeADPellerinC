import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import importlib
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer, PowerTransformer

def data_collection(file_path, time_col):
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
    
file_path = "C:/Users/pellerinc/Downloads/occupancy+detection/datatest2.txt"
APLfile_path = "C:/Users/pellerinc/TimeSeAD-master/Daten_von_APL/Export/i.O/WLTC_1.csv"
#df1 = read_with_time_index(file_path,'date')
df2 = data_collection("C:/Users/pellerinc/TimeSeAD-master/Daten_von_APL/Export/i.O/WLTC_1.csv", 's') # In Ordnung
dftest = data_collection("C:/Users/pellerinc/TimeSeAD-master/Daten_von_APL/Export/i.O/WLTC_8.csv", 's') # In Ordnung
df3 = data_collection("C:/Users/pellerinc/TimeSeAD-master/Daten_von_APL/Export/n.i.O/WLTC_118.csv", 's') # Nicht in Ordnung
df4 = data_collection("C:/Users/pellerinc/TimeSeAD-master/Daten_von_APL/Export/n.i.O/WLTC_119.csv", 's') # Nicht in Ordnung
df5 = data_collection("C:/Users/pellerinc/TimeSeAD-master/Daten_von_APL/Export/n.i.O/WLTC_120.csv", 's') # Nicht in Ordnung
df6 = data_collection("C:/Users/pellerinc/TimeSeAD-master/Daten_von_APL/Export/n.i.O/WLTC_121.csv", 's') # Nicht in Ordnung
df7 = data_collection("C:/Users/pellerinc/TimeSeAD-master/Daten_von_APL/Export/n.i.O/WLTC_122.csv", 's') # Nicht in Ordnung

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

print(df_empty_columns)

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

class MethodChooser:
    """
    Classe responsable de l'analyse du dataset et de la sélection automatique du modèle.
    """

    @staticmethod
    def analyze_dataset(data, labels=None):
        """Analyse le dataset pour extraire les caractéristiques principales."""
        num_samples, num_features = data.shape
        labels_available = labels is not None and len(labels) == num_samples

        # Calcul du ratio d'anomalies (si les labels existent)
        if labels_available and len(labels) > 0:
            anomaly_ratio = np.sum(labels) / len(labels)
        else:
            anomaly_ratio = 0

        # Déterminer le type d'anomalie
        anomaly_type = (
            "point" if anomaly_ratio < 0.01 else
            "contextual" if anomaly_ratio < 0.05 else
            "collective"
        ) if labels_available else "unknown"

        # Vérifier si le dataset est en temps réel (grande taille et plusieurs features)
        real_time = num_samples > 10_000 and num_features > 5

        return {
            "num_samples": num_samples,
            "num_features": num_features,
            "labels_available": labels_available,
            "anomaly_type": anomaly_type,
            "real_time": real_time
        }

    @staticmethod
    def select_best_model(data, labels_available=False, anomaly_type="point", real_time=False, forced_model=None):
        """Sélectionne et importe automatiquement le modèle optimal."""
        if forced_model:
            selected_model = forced_model
        else:
            num_samples, num_features = data.shape

            model_choices = {
                "point": "lstm_ae" if not labels_available else "gdn",
                "contextual": "anom_trans" if not labels_available else "mtad_gat",
                "collective": "mscred" if not labels_available else "thoc",
            }

            if real_time:
                selected_model = "gdn" if labels_available else "tcn_prediction"
            elif num_samples < 1000:
                selected_model = "donut" if labels_available else "tcn_prediction"
            elif num_samples > 10_000:
                selected_model = "anom_trans" if not labels_available else "mtad_gat"
            else:
                selected_model = model_choices.get(anomaly_type, "omni_anomaly" if not labels_available else "lstm_prediction")

        # 🔹 Trouver le chemin du modèle dans MODEL_MAP
        model_path = None
        for category, models in MODEL_MAP.items():
            if selected_model in models:
                model_path = models[selected_model]
                break

        if not model_path:
            raise ValueError(f" Modèle {selected_model} introuvable dans MODEL_MAP !")

        # 🔹 Importation dynamique du modèle
        module_name, class_name = model_path.rsplit(".", 1)  # Séparer module et classe
        try:
            module = importlib.import_module(module_name)
            model_class = getattr(module, class_name)
            print(f" Modèle sélectionné : {selected_model} ({class_name})")

            #  Vérifier si le modèle sélectionné a des paramètres spécifiques
            if class_name == "TCNPredictionAnomalyDetector":
                # Importation dynamique de TCNPrediction
                prediction_module = importlib.import_module("timesead.models.prediction.tcn_prediction")
                TCNPrediction = getattr(prediction_module, "TCNPrediction")

                input_dim = data.shape[1]
                window_size = 50  # Ajuste selon ton pipeline
                base_model = TCNPrediction(input_dim=input_dim, window_size=window_size)

                model_instance = model_class(base_model)
                print(f" {class_name} instancié avec input_dim={input_dim}, window_size={window_size}")
                return model_instance

            elif class_name == "AnomalyTransformer":
                input_dim = data.shape[1]
                win_size = 50  # Ajuste selon ton pipeline

                model_instance = model_class(win_size=win_size, input_dim=input_dim)
                print(f" {class_name} instancié avec win_size={win_size}, input_dim={input_dim}")
                return model_instance

            #  Pour les modèles qui n'ont pas besoin de paramètres spécifiques
            return model_class()

        except Exception as e:
            raise ImportError(f" Erreur lors de l'importation de {selected_model}: {e}")
            
class UpstreamDataPreparation:
    def __init__(self, method=None, threshold=None, seq_length=None, scale_method=None, interpolation_method =None):
        self.method = method  # Méthode de détection des outliers (IQR ou MAD)
        self.threshold = threshold  # Seuil pour détecter les outliers
        self.seq_length = seq_length  # Longueur des séquences temporelles
        self.scale_method = scale_method  # Méthode de normalisation
        self.interpolation_method = interpolation_method
        self.scaler = None  # Scaler à réutiliser

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
    
        if scale_method == "minmax":
            self.scaler = MinMaxScaler(feature_range=(0, 1))
        elif scale_method == "standard":
            self.scaler = StandardScaler()
        elif scale_method == "robust":
            self.scaler = RobustScaler()
        elif scale_method == "quantile":
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
    
    def process(self, data, model_name, train_size=0.8):
        """
        Pipeline complet : suppression des erreurs capteurs, détection anomalies moteur,
        normalisation et conversion en séquences temporelles.
        """
        data_clean, anomalies_motor = self.preprocess_data(data)
        
        print("Normalisation des données...")
        train_data = data_clean.iloc[:int(len(data_clean) * train_size)]
        test_data = data_clean.iloc[int(len(data_clean) * train_size):]
        train_data, test_data = self.normalize_data(train_data, test_data)
        
        X_train_tensor = self.create_sequences(train_data)
        X_test_tensor = self.create_sequences(test_data)

        torch.save(X_train_tensor, "X_train_tensor.pt")
        torch.save(X_test_tensor, "X_test_tensor.pt")
        print("Tenseurs sauvegardés avec succès dans 'X_train_tensor.pt' et 'X_test_tensor.pt'.")
        
        return X_train_tensor, X_test_tensor, anomalies_motor
        