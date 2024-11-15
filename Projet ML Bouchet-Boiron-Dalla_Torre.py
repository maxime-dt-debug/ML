#!/usr/bin/env python
# coding: utf-8

# In[16]:


# Importation des bibliothèques pour manipulation de données et visualisation
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[20]:


# Chemin vers le dossier contenant les fichiers CSV (remplacez par votre chemin)
data_folder = r"C:\Titres MachineLearning"

# Liste pour stocker les DataFrames de chaque fichier
all_data = []

# Parcourir tous les fichiers CSV dans le dossier
for file in os.listdir(data_folder):
    if file.endswith(".csv"):  # Vérifie que le fichier est un CSV
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

# Vérifier les premières lignes du DataFrame combiné
print(data.head())


# In[21]:


# Convertir la colonne 'date' en format datetime
data['date'] = pd.to_datetime(data['date'], format='%d/%m/%Y')

# Nettoyer et convertir la colonne 'change_percent'
# - Supprimer le caractère "%" et remplacer les virgules par des points
data['change_percent'] = data['change_percent'].astype(str)  # Convertir en chaînes de caractères
data['change_percent'] = data['change_percent'].str.replace("%", "").str.replace(",", ".")
data['change_percent'] = data['change_percent'].replace("", "0").astype(float) / 100  # Convertir en float

# Vérifier les types de données après conversion
print(data.dtypes)

# Afficher un aperçu des données nettoyées
print(data.head())


# In[22]:


# Trier les données par action et par date
data = data.sort_values(by=["Action", "date"]).reset_index(drop=True)

# Vérifier que les données sont bien triées
print(data.head())


# In[26]:


# Vérifier les valeurs manquantes
print(data.isnull().sum())


# In[23]:


# Créer un histogramme pour observer la distribution des variations
sns.histplot(data['change_percent'], kde=True, bins=50)
plt.title("Distribution des variations (%)")
plt.xlabel("Variation (%)")
plt.ylabel("Fréquence")
plt.show()


# In[ ]:



     


# In[ ]:





# In[ ]:




