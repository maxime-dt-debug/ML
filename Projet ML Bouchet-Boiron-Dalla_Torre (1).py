#!/usr/bin/env python
# coding: utf-8

# In[33]:


# Importation des bibliothèques nécessaires
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# In[34]:


# Chemin vers le dossier contenant les fichiers CSV
data_folder = r"C:\Titres MachineLearning"

# Liste pour stocker les DataFrames de chaque fichier
all_data = []

# Parcourir tous les fichiers CSV dans le dossier
for file in os.listdir(data_folder):
    if file.endswith(".csv") and file != "cleaned_data.csv":  # Ignorer cleaned_data.csv
        # Charger uniquement les colonnes nécessaires : Date et Variation %
        filepath = os.path.join(data_folder, file)
        df = pd.read_csv(
            filepath,
            delimiter=",",         # Séparateur des colonnes
            usecols=["Date", "Variation %"],  # Charger uniquement les colonnes nécessaires
            decimal=",",           # Gérer les virgules comme séparateurs décimaux
        )
        
        # Renommer les colonnes pour simplifier leur manipulation
        df.rename(columns={"Date": "date", "Variation %": "change_percent"}, inplace=True)
        
        # Ajouter une colonne pour identifier l'action (nom du fichier sans extension)
        df['Action'] = os.path.splitext(file)[0]
        
        # Ajouter le DataFrame nettoyé à la liste
        all_data.append(df)

# Combiner tous les fichiers en un seul DataFrame
data = pd.concat(all_data, ignore_index=True)


# In[35]:


# Convertir la colonne 'date' en format datetime
data['date'] = pd.to_datetime(data['date'], format='%d/%m/%Y')

# Nettoyer et convertir la colonne 'change_percent'
data['change_percent'] = data['change_percent'].astype(str)  # Convertir en chaînes de caractères
data['change_percent'] = data['change_percent'].str.replace("%", "").str.replace(",", ".")
data['change_percent'] = data['change_percent'].replace("", "0").astype(float) / 100  # Convertir en float

# Trier les données par action et par date
data = data.sort_values(by=["Action", "date"]).reset_index(drop=True)


# In[36]:


# Vérifier un aperçu des données nettoyées
print(data.head())

# Vérifier les types de données
print(data.dtypes)

# Vérifier les valeurs manquantes
print(data.isnull().sum())


# In[37]:


# Spliter les données par action (80% Entraînement / 20% Test)
splits = {}

for action in data['Action'].unique():
    # Sous-ensemble pour une action spécifique
    subset = data[data['Action'] == action].reset_index(drop=True)
    
    # Taille de l'ensemble d'entraînement
    train_size = int(len(subset) * 0.8)
    
    # Division en ensembles d'entraînement et de test
    train = subset[:train_size]
    test = subset[train_size:]
    
    # Stocker les ensembles dans un dictionnaire
    splits[action] = {
        "train": train,
        "test": test
    }

# Exemple de vérification pour une action spécifique
example_action = list(splits.keys())[0]
print(f"Exemple pour l'action : {example_action}")
print("Ensemble d'entraînement :")
print(splits[example_action]["train"].head())
print("Ensemble de test :")
print(splits[example_action]["test"].head())


# In[38]:


# Modèle Ridge avec régularisation
model = Ridge(alpha=1.0)

# Effectuer la validation croisée et entraîner le modèle pour chaque action
evaluation_results = {}

for action in splits.keys():
    # Obtenir les ensembles d'entraînement et de test
    train = splits[action]["train"]
    test = splits[action]["test"]
    
    # Définir X et y pour l'entraînement et le test
    X_train = np.arange(len(train)).reshape(-1, 1)
    y_train = train['change_percent']
    X_test = np.arange(len(train), len(train) + len(test)).reshape(-1, 1)
    y_test = test['change_percent']
    
    # Effectuer une validation croisée
    tscv = TimeSeriesSplit(n_splits=5)
    scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring='neg_mean_squared_error')
    
    # Entraîner le modèle
    model.fit(X_train, y_train)
    
    # Faire les prédictions
    predictions = model.predict(X_test)
    
    # Calculer les métriques
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)
    
    # Stocker les résultats
    evaluation_results[action] = {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
        "Validation_Cross_Val_Score": scores
    }

# Afficher les résultats pour toutes les actions
for action, metrics in evaluation_results.items():
    print(f"\nRésultats pour {action}:")
    for metric, value in metrics.items():
        if isinstance(value, np.ndarray):  # Pour les scores de validation croisée
            print(f"{metric}: {value}")
        else:
            print(f"{metric}: {value:.5f}")


# In[39]:


# Résidus pour une action spécifique
action_to_plot = example_action
train = splits[action_to_plot]["train"]
test = splits[action_to_plot]["test"]

X_train = np.arange(len(train)).reshape(-1, 1)
y_train = train['change_percent']
X_test = np.arange(len(train), len(train) + len(test)).reshape(-1, 1)
y_test = test['change_percent']

# Entraîner le modèle pour cette action
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Résidus (erreurs)
residuals = y_test - predictions

# Visualiser les résidus
plt.figure(figsize=(10, 5))
plt.scatter(test['date'], residuals, color='purple', alpha=0.7)
plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.title(f"Résidus pour {action_to_plot}")
plt.xlabel("Date")
plt.ylabel("Résidus (erreurs)")
plt.show()

     


# In[ ]:





# In[ ]:




