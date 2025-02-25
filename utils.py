import mlflow
import mlflow.sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics


def preprocess_data_with_mlflow(app_train, app_test):
    # Démarrer une expérience MLflow
    mlflow.set_experiment("risque de crédit")
    
    with mlflow.start_run():
        # Enregistrement des paramètres
        mlflow.log_param("imputer_strategy", "median")
        mlflow.log_param("scaler_range", "(0, 1)")

        # Suppression de la cible (TARGET) des données d'entraînement
        if 'TARGET' in app_train:
            train = app_train.drop(columns = ['TARGET'])
        else:
            train = app_train.copy()

        # Création de la copie des données de test
        test = app_test.copy()

        # Imputation des valeurs manquantes (stratégie de la médiane)
        imputer = SimpleImputer(strategy = 'median')
        imputer.fit(train)
        train = imputer.transform(train)
        test = imputer.transform(app_test)

        # Mise à l'échelle des caractéristiques (plage [0, 1])
        scaler = MinMaxScaler(feature_range = (0, 1))
        scaler.fit(train)
        train = scaler.transform(train)
        test = scaler.transform(test)

        # Enregistrer les modèles (imputeur et scaler) en tant qu'artefacts dans MLflow
        mlflow.sklearn.log_model(imputer, "imputer_model")
        mlflow.sklearn.log_model(scaler, "scaler_model")

        # Enregistrement des métriques
        mlflow.log_metric("train_shape", train.shape[0])
        mlflow.log_metric("test_shape", test.shape[0])

        # Affichage des dimensions finales des données
        print('Training data shape: ', train.shape)
        print('Testing data shape: ', test.shape)
    
    return train, test

from imblearn.over_sampling import SMOTE

def apply_smote_with_mlflow(X, y):
    """
    Applique SMOTE pour équilibrer les classes dans un jeu de données déséquilibré.
    Intègre le suivi avec MLflow.
    
    :param X: Features (données d'entraînement sans la cible)
    :param y: Cible (labels)
    :return: X_smote, y_smote (données équilibrées)
    """
    # Initialiser SMOTE
    smote = SMOTE(random_state=42)
    
    # Démarrer une expérience MLflow
    mlflow.set_experiment("risque de crédit")
    
    with mlflow.start_run():
        # Enregistrer les paramètres SMOTE
        mlflow.log_param("SMOTE_strategy", "auto")
        mlflow.log_param("SMOTE_random_state", 42)
        
        imputer = SimpleImputer(strategy="median")
        X_imputed = imputer.fit_transform(X)
        # Application de SMOTE
        X_smote, y_smote = smote.fit_resample(X_imputed, y)
        
        # Enregistrer des métriques sur la distribution des classes
        original_distribution = dict(zip(*np.unique(y, return_counts=True)))
        balanced_distribution = dict(zip(*np.unique(y_smote, return_counts=True)))
        
        mlflow.log_metric("Original_class_0", original_distribution.get(0, 0))
        mlflow.log_metric("Original_class_1", original_distribution.get(1, 0))
        mlflow.log_metric("Balanced_class_0", balanced_distribution.get(0, 0))
        mlflow.log_metric("Balanced_class_1", balanced_distribution.get(1, 0))
        
        # Afficher les distributions pour vérification
        print("Original class distribution:", original_distribution)
        print("Balanced class distribution:", balanced_distribution)
        
    return X_smote, y_smote


# evaluation.py




# Définition des fonctions
def fonction_metier_avec_penalite_faux_negatifs(y_true, y_pred):
    """
    Calculer un score métier avec une pénalité renforcée pour les faux négatifs.
    :param y_true: Array des valeurs vraies
    :param y_pred: Array des valeurs prédites
    :return: Gain (score métier)
    """
    TP_coeff = 1          # Vrais positifs
    FP_coeff = 0       # Faux positifs
    FN_coeff = -20        # Faux négatifs (fortement pénalisé)
    TN_coeff = 1        # Vrais négatifs

    TN, FP, FN, TP = metrics.confusion_matrix(y_true, y_pred).ravel()
    gain = (TP * TP_coeff + TN * TN_coeff + FP * FP_coeff + FN * FN_coeff) / (TP + TN + FP + FN)
    return gain

def score_metier_max(y_pred_proba, y_true, verbose=True):
    """
    Trouver le score métier maximum en fonction du threshold.
    :param y_pred_proba: Probabilités prédites par le modèle
    :param y_true: Valeurs vraies
    :param verbose: Afficher le graphe si True
    :return: Score métier maximal
    """
    scores = []
    thresholds = np.linspace(0, 1, num=101)
    for threshold in thresholds:
        y_pred = np.where(y_pred_proba > threshold, 1, 0)
        score = fonction_metier_avec_penalite_faux_negatifs(y_true, y_pred)
        scores.append(score)

    if verbose:
        score_max = max(scores)
        opti_threshold = thresholds[scores.index(score_max)]
        plt.figure(figsize=(6, 5))
        plt.plot(thresholds, scores, label="Model Score")
        plt.axvline(x=opti_threshold, color='k', linestyle='--', label=f"Optimal Threshold: {opti_threshold:.2f}")
        plt.title("Score métier en fonction du threshold")
        plt.xlabel("Threshold")
        plt.ylabel("Score métier")
        plt.legend()
        plt.show()
        print(f"Score métier maximum: {score_max:.2f}")
        print(f"Threshold optimal: {opti_threshold:.2f}")
    else:
        return max(scores)

def conf_mat_transform(y_true, y_pred):
    """
    Afficher la matrice de confusion.
    :param y_true: Valeurs vraies
    :param y_pred: Valeurs prédites
    :return: None (Affiche un heatmap de la matrice de confusion)
    """
    conf_mat = metrics.confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_mat, annot=True, cmap="Blues", fmt='d')
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.title("Matrice de confusion")
    plt.show()

def eval_score(model, X_val, y_true, seuil=0.5):
    """
    Évaluer un modèle sur différentes métriques et afficher les résultats.
    :param model: Modèle à évaluer
    :param X_val: Données de validation
    :param y_true: Valeurs vraies
    :param seuil: Seuil pour les prédictions
    :return: Différentes métriques et les probabilités prédites
    """

    # Imputation des valeurs manquantes
    imputer = SimpleImputer(strategy="median")
    
    X_val_imputed = imputer.fit_transform(X_val)
    
    # Prédiction des probabilités
    y_pred_proba = model.predict_proba(X_val_imputed)[:, 1]
    
    # Conversion des probabilités en classes binaires (0 ou 1)
    y_pred = np.where(y_pred_proba > seuil, 1, 0)
    
    # Calcul du score métier
    metier = fonction_metier_avec_penalite_faux_negatifs(y_true, y_pred)
    
    # Calcul des métriques classiques
    accuracy = metrics.accuracy_score(y_true, y_pred)
    precision = metrics.precision_score(y_true, y_pred)
    recall = metrics.recall_score(y_true, y_pred)
    f1_score = metrics.f1_score(y_true, y_pred)
    fbeta_score = metrics.fbeta_score(y_true, y_pred, beta=2)
    roc_auc = metrics.roc_auc_score(y_true, y_pred_proba)
    
    # Affichage des résultats
    print(f"Score métier: {metier:.2f}")
    print(f"Accuracy score: {accuracy:.2f}")
    print(f"Precision score: {precision:.2f}")
    print(f"Recall score: {recall:.2f}")
    print(f"F1 score: {f1_score:.2f}")
    print(f"Fbeta score: {fbeta_score:.2f}")
    print(f"ROC AUC score: {roc_auc:.2f}")
    
    # Affichage de la matrice de confusion
    conf_mat_transform(y_true, y_pred)
    
    # Courbe ROC
    plt.figure(figsize=(6, 5))
    fpr, tpr, _ = metrics.roc_curve(y_true, y_pred_proba)
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], 'r--')
    plt.title("Courbe ROC")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.show()
    
    return metier, accuracy, precision, recall, f1_score, fbeta_score, roc_auc, y_pred_proba
