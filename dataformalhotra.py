import torch
from mycodefiles.module.datapreparation import UpstreamDataPreparation

def get_prepared_data():
    # Initialisation de l'objet de préparation avec les paramètres définis
    udp = UpstreamDataPreparation(
        method="IQR", 
        seq_length=50, 
        scale_method="standard", 
        interpolation_method="linear", 
        window_params={"window_size": 50, "step_size": 5, "reverse": False, "prediction_horizon": 3}
    )

    # Chargement des données depuis le fichier CSV
    df = udp.data_collection(
        'C:/Users/pellerinc/TimeSeAD-master/Daten_von_APL/Export/i.O/WLTC_1.csv',
        time_col='s'
    )

    # Prétraitement : détection des anomalies moteur et interpolation
    df_clean, _ = udp.preprocess_data(df)

    # Transformation en jeux de données pipeline TimeSeAD (train/test)
    train_pipeline, val_pipeline, _ = udp.process(df_clean, train_size=0.8, use_pipeline=True)

    return train_pipeline, val_pipeline
    
    
