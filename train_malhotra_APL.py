import torch
import time
import matplotlib.pyplot as plt
import random
import os
import numpy as np

from timesead.models.prediction import LSTMPrediction, LSTMPredictionAnomalyDetector
from timesead.optim.trainer import EarlyStoppingHook
from timesead_experiments.utils import data_ingredient, load_dataset, training_ingredient, train_model, make_experiment, \
    make_experiment_tempfile, serialization_guard, get_dataloader
from timesead.utils.utils import Bunch
from torch.utils.data import Dataset
from timesead.optim.loss import LogCoshLoss
#from dataformalhotra import get_prepared_data
from mycodefiles.module.datapreparation import UpstreamDataPreparationiO, UpstreamDataPreparationForTest
from timesead_experiments.utils.experiment_functions import SerializationGuard
from timesead.plots.dataset_plots import plot_features_against_anomaly
from timesead.utils.plot_utils import plot_sequence_against_anomaly

PREDICTION_HORIZON = 5

def get_prepared_data():
    # Initialisation de l'objet de préparation avec les paramètres définis
    udp = UpstreamDataPreparationiO(
        method="IQR",
        seq_length=35,
        scale_method="standard",
        interpolation_method="linear",
        window_params={"window_size": 30, "step_size": 5, "reverse": False, "prediction_horizon": PREDICTION_HORIZON}
    )

    # Chargement des données depuis le fichier CSV
    df = udp.data_collection(
        directory='C:/Users/pellerinc/TimeSeAD-master/Daten_von_APL/Export/i.O',
        time_col='s',
        max_concatenate=6
    )

    # Prétraitement : détection des anomalies moteur et interpolation
    # df_clean, _ = udp.preprocess_data(df) (REDONDANCE AVEC PROCESS)

    #print("Shape du DataFrame nettoyé :", df_clean.shape)

    # Transformation en jeux de données pipeline TimeSeAD (train/test)
    train_pipeline, val_pipeline = udp.process(df, train_size=0.8, use_pipeline=True)

    if len(train_pipeline) == 0 or len(val_pipeline) == 0:
        raise ValueError("Le dataset d'entraînement ou de validation est vide. Vérifiez les paramètres de fenêtrage.")

    return train_pipeline, val_pipeline, udp

'''    
def get_prepared_niO_data():
    # Initialisation de l'objet de préparation avec les paramètres définis
    udp = UpstreamDataPreparationForTest(
        method="IQR",
        seq_length=33,
        scale_method="standard",
        interpolation_method="linear",
        window_params={"window_size": 30, "step_size": 5, "reverse": False, "prediction_horizon": PREDICTION_HORIZON}
    )

    # Chargement des données depuis le fichier CSV
    df = udp.data_collection(
        directory='C:/Users/pellerinc/TimeSeAD-master/Daten_von_APL/Export/i.O',
        time_col='s',
        max_concatenate=2
    )

    # Prétraitement : détection des anomalies moteur et interpolation
    df_clean, _ = udp.preprocess_data(df)

    print("Shape du DataFrame nettoyé :", df_clean.shape)

    # Transformation en jeux de données pipeline TimeSeAD (train/test)
    train_pipeline, val_pipeline, _ = udp.process(df_clean, train_size=0.8, use_pipeline=True)

    if len(train_pipeline) == 0 or len(val_pipeline) == 0:
        raise ValueError("Le dataset d'entraînement ou de validation est vide. Vérifiez les paramètres de fenêtrage.")

    return train_pipeline, val_pipeline
'''

class LabeledWindowDataset(Dataset):
    def __init__(self, base_dataset, predicted_anomaly_labels):
        self.anomaly_labels = predicted_anomaly_labels
        self.base_dataset = torch.utils.data.Subset(base_dataset, list(range(len(predicted_anomaly_labels))))

    def __len__(self):
        return len(self.anomaly_labels)

    def __getitem__(self, idx):
        data_tuple, _ = self.base_dataset[idx]
        label = self.anomaly_labels[idx]
        label_tensor = torch.tensor([label] * data_tuple[0].shape[-1])
        return (data_tuple, label_tensor)

    
def plot_mahalanobis_score_per_feature_per_timestep(detector, dataloader, features=3, file_id=None, window_params = {"window_size": 30, "step_size": 5}, directory =""):
    detector.model.eval()
    all_errors = []

    with torch.no_grad():
        for b_inputs, b_targets in dataloader:
            x = b_inputs[0].to(detector.dummy.device)  # (T, B, D)
            y_true = b_targets[-1].to(detector.dummy.device)  # (H, B, D)
            y_pred = detector.model((x,))
            error = y_true - y_pred  # (H, B, D)
            all_errors.append(error.permute(1, 0, 2).cpu())  # (B, H, D)

    errors = torch.cat(all_errors, dim=0)  # (N, H, D)
    errors_centered = errors.reshape(-1, errors.shape[1] * errors.shape[2]) - detector.mean.cpu()
    precision = detector.precision.cpu()

    D = errors.shape[2]
    H = errors.shape[1]

    if file_id:
        dir_path = os.path.join(directory, file_id) #A REMPLACER SELON LE TYPE DE FICHIER ETC
        os.makedirs(dir_path, exist_ok=True)
    step_size = window_params.get("step_size", 1)
    timestep = 0.1

    for f in range(min(D, features)):
        plt.figure(figsize=(12, 4))
        feature_indices = [h * D + f for h in range(H)]
        feature_errors = errors_centered[:, feature_indices]
        precision_sub = precision[feature_indices][:, feature_indices]
        score = torch.einsum('bi,ij,bj->b', feature_errors, precision_sub, feature_errors)
        
        time_axis = np.arange(len(score)) * step_size * timestep
        
        # 🔺 Seuil basé sur le quantile 99%
        threshold = np.quantile(score, 0.99)
        plt.axhline(threshold, color='red', linestyle='--', label=f"Threshold (99th percentile)")
        plt.plot(time_axis, score, label=f"Mahalanobis Score – Feature {f+1}")
        plt.xlabel("Time in seconds")
        plt.ylabel("Mahalanobis Score")
        plt.title(f"Anomaly score for feature {f+1} : {file_id}")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        if file_id:
            save_path = os.path.join(dir_path, f"feature_{f+1}.png")
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

experiment = make_experiment(ingredients=[data_ingredient, training_ingredient])


'''
def get_training_pipeline():
    return {
        'prediction': {'class': 'PredictionTargetTransform', 'args': {'window_size': 50, 'prediction_horizon': 3,
                                                                      'replace_labels': True}}
    }


def get_test_pipeline():
    return {
        'prediction': {'class': 'PredictionTargetTransform', 'args': {'window_size': 50, 'prediction_horizon': 3,
                                                                      'replace_labels': False}}
    }
'''

def get_batch_dim():
    return 1


@data_ingredient.config
def data_config():
    #pipeline = get_training_pipeline()

    ds_args = dict(
        training=True,
    )

    split = (0.8, 0.2)


@training_ingredient.config
def training_config():
    batch_dim = get_batch_dim()
    loss = [torch.nn.MSELoss] #LogCoshLoss
    trainer_hooks = [
        ('post_validation', EarlyStoppingHook)
    ]
    scheduler = {
        'class': torch.optim.lr_scheduler.MultiStepLR,
        'args': dict(milestones=[20], gamma=0.1)
    }

def fix_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@experiment.config
def config():
    # Model-specific parameters
    model_params = dict(
        lstm_hidden_dims=[64, 64, 32], #][50, 30], [30, 20]
        linear_hidden_layers=[], #[],[50], [[100, 50]]
        linear_activation=[torch.nn.ELU()], #, torch.nn.ReLU()
    )

    train_detector = True
    save_detector = True
    seed = 42


@experiment.command(unobserved=True)
@serialization_guard
def get_datasets():
    train_ds, val_ds, udp = get_prepared_data() #initialement train_ds, val_ds = load_dataset()
    print("Length of training dataset:", len(train_ds))
    print("Length of validation dataset:", len(val_ds))

    return get_dataloader(train_ds), get_dataloader(val_ds)


@experiment.command(unobserved=True)
@serialization_guard('model', 'val_loader')
def get_anomaly_detector(model, val_loader, training, _run, save_detector=True):
    training = Bunch(training)

    if val_loader is None:
        train_ds, val_ds = get_prepared_data() #initialement train_ds, val_ds = load_dataset()
        print("Length of training dataset:", len(train_ds))
        print("Length of validation dataset:", len(val_ds))
        # Train for 0 epochs to get the val loader
        trainer = train_model(_run, model, train_ds, val_ds, epochs=0)
        val_loader = trainer.val_iter

    detector = LSTMPredictionAnomalyDetector(model).to(training.device)
    detector.fit(val_loader)

    if save_detector:
        with make_experiment_tempfile('final_model.pth', _run, mode='wb') as f:
            torch.save(dict(detector=detector), f)

    return detector


@experiment.automain
@serialization_guard
def main(model_params, dataset, training, _run, train_detector=True, load_detector=False, seed=42):
    fix_random_seed(seed)
    total_start_time = time.time()
    ds_params = Bunch(dataset)
    train_params = Bunch(training)
    cached_detector_path = os.path.join(_run.observers[0].dir, "final_model.pth")

    udp = None
    trainer = None

    if load_detector and os.path.exists(cached_detector_path):
        print("📦 Chargement du modèle détecteur depuis le cache...")
        checkpoint = torch.load(cached_detector_path, map_location=train_params.device)
        detector = checkpoint["detector"]
        model = detector.model
    else:
        # 📥 Chargement des données
        train_ds, val_ds, udp = get_prepared_data()
        print("Length of training dataset:", len(train_ds))
        print("Length of validation dataset:", len(val_ds))

        # 🧠 Instanciation du modèle
        model = LSTMPrediction(train_ds.num_features,
                               prediction_horizon=PREDICTION_HORIZON,
                               **model_params).to(train_params.device)

        # 🎯 Entraînement
        trainer = train_model(_run, model, train_ds, val_ds)
        early_stop = trainer.hooks['post_validation'][-1]
        model = early_stop.load_best_model(trainer, model, train_params.epochs)

        # 🔍 Détection d'anomalies (Malhotra)
        detector = get_anomaly_detector(model, trainer.val_iter)
        if isinstance(detector, SerializationGuard):
            detector = detector.object

    if train_detector:
        total_end_time = time.time()
        print(f"🏁 Temps total d'exécution : {total_end_time - total_start_time:.2f} secondes.")

        if udp is not None:
            udp_niO = UpstreamDataPreparationForTest(
                scaler=udp.scaler,
                window_params=udp.window_params
            )

        # TEST FICHIER n.i.O
        for i in range(118, 124):
            print(f"\n📂 Traitement du fichier WLTC_{i}.csv...")
            try:
                df_nio = udp_niO.data_collection_niO(
                    directory="C:/Users/pellerinc/TimeSeAD-master/Daten_von_APL/Export/n.i.O",
                    time_col="s",
                    file_index=i
                )
                df_nio = udp_niO.prepare_test_dataframe(df_nio)
                test_pipeline = udp_niO.process(df_nio)
                test_loader = get_dataloader(test_pipeline)

                #print(f"📊 Score d’anomalie – WLTC_{i}.csv")
                plot_mahalanobis_score_per_feature_per_timestep(detector, 
                test_loader, 
                features=49, 
                file_id=f"WLTC_{i}", 
                window_params = udp_niO.window_params, 
                directory = "C:/Users/pellerinc/TimeSeAD-master/malhotra_visualization/anomaly_scores_n.i.O"
                )
          

            except FileNotFoundError:
                print(f"❌ Fichier WLTC_{i}.csv introuvable.")
            except Exception as e:
                print(f"⚠️ Erreur sur WLTC_{i}.csv : {e}")
               
        # TEST FICHIERS i.O AVEC udp_niO
        for i in range(1, 7):
            print(f"\n📂 Traitement du fichier WLTC_{i}.csv (i.O)...")
            try:
                df_io = udp_niO.data_collection_niO(
                    directory="C:/Users/pellerinc/TimeSeAD-master/Daten_von_APL/Export/i.O",
                    time_col="s",
                    file_index=i
                )
                df_io = udp_niO.prepare_test_dataframe(df_io)
                test_pipeline = udp_niO.process(df_io)
                test_loader = get_dataloader(test_pipeline)

                plot_mahalanobis_score_per_feature_per_timestep(
                    detector,
                    test_loader,
                    features=49,
                    file_id=f"WLTC_{i}_iO",
                    window_params=udp_niO.window_params,
                    directory="C:/Users/pellerinc/TimeSeAD-master/malhotra_visualization/anomaly_scores_i.O"
                )

            except FileNotFoundError:
                print(f"❌ Fichier WLTC_{i}.csv (i.O) introuvable.")
            except Exception as e:
                print(f"⚠️ Erreur sur WLTC_{i}.csv (i.O) : {e}")


    '''
    # 🎨 Courbes de loss
    try:
        loss_keys = sorted([k for k in _run.info if k.startswith("train_loss") or k.startswith("val_loss")])
        for key in loss_keys:
            plt.plot(_run.info[key], label=key)

        if loss_keys:
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Courbes de loss (entraînement/validation)")
            plt.legend()
            plt.grid()
            plt.tight_layout()
            plt.show()
    except Exception as e:
        print("⚠️ Impossible d'afficher les courbes:", e)
    '''

    return dict(detector=detector, model=model)
