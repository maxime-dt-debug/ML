import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



# Chemin vers le dossier contenant les fichiers CSV
data_folder = r"C:\Titres MachineLearning"

#  stock les DataFrames de chaque fichier
all_data = []

for file in os.listdir(data_folder):
    if file.endswith(".csv"):  #prend tous les csv du fichier
        filepath = os.path.join(data_folder, file)
        df = pd.read_csv(
            filepath,
            delimiter=",", #pour delimiter les colonnes csv car delimiter =, 
            usecols=["Date", "Variation %"],  # charge colonnes necessaires
            decimal=",",          
        )
        
        # renommer les colonnes
        df.rename(columns={"Date": "date", "Variation %": "change_percent"}, inplace=True)
        
       
        df['Action'] = os.path.splitext(file)[0]  # aidentifie l'action
        
     
        all_data.append(df)


data = pd.concat(all_data, ignore_index=True) # combine tous les fichiers en un seul DataFrame


data['date'] = pd.to_datetime(data['date'], format='%d/%m/%Y') # convertiz les dates en datetime


data['change_percent'] = data['change_percent'].astype(str)  # convertie en chaines de caracteres
data['change_percent'] = data['change_percent'].str.replace("%", "").str.replace(",", ".")
data['change_percent'] = data['change_percent'].replace("", "0").astype(float) / 100  # en float


data = data.sort_values(by=["Action", "date"]).reset_index(drop=True) # Trie par action et par date


#on v�rfie pour voir si les donn�es on bien �t� tri�
print(data.head())


print(data.dtypes)

print(data.isnull().sum()) # si il manque des valeurs





# splite les donnees par action 80 entrainement 20 test
splits = {}

for action in data['Action'].unique():

    subset = data[data['Action'] == action].reset_index(drop=True)
    
    # taille de l'ensemble d'entrainement
    train_size = int(len(subset) * 0.8)
    
    # division en ensembles d'entra�nement et de test
    train = subset[:train_size]
    test = subset[train_size:]
    
    # stocke les ensembles dans un dictionnaire
    splits[action] = {
        "train": train,
        "test": test
    }

#v�rification pour une action 
example_action = list(splits.keys())[0]
print(f"Exemple pour l'action : {example_action}")
print("Ensemble d'entra�nement :")
print(splits[example_action]["train"].head())
print("Ensemble de test :")
print(splits[example_action]["test"].head())


# Mod�le ridge avec r�gularisation
model = Ridge(alpha=1.0)

# effectue la validation crois�e et entra�ner le mod�le pour chaque action
evaluation_results = {}

for action in splits.keys():
    # ensembles d'entrainement et de test
    train = splits[action]["train"]
    test = splits[action]["test"]
    
    # definir X et y pour lentrainement et le test
    X_train = np.arange(len(train)).reshape(-1, 1)
    y_train = train['change_percent']
    X_test = np.arange(len(train), len(train) + len(test)).reshape(-1, 1)
    y_test = test['change_percent']
    
    # effectuer une validation croisee
    tscv = TimeSeriesSplit(n_splits=5)
    scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring='neg_mean_squared_error')
    
    # entraine
    model.fit(X_train, y_train)
    
    # pr�ditions
    predictions = model.predict(X_test)
    
    # Calcul metriques
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)
    
    # stocke les resultats
    evaluation_results[action] = {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
        "Validation_Cross_Val_Score": scores
    }

# affiche les resultats pour toutes les actions
for action, metrics in evaluation_results.items():
    print(f"\nR�sultats pour {action}:")
    for metric, value in metrics.items():
        if isinstance(value, np.ndarray):  # Pour les scores de validation croisee
            print(f"{metric}: {value}")
        else:
            print(f"{metric}: {value:.5f}")




#  pour une action sp�cifique
action_to_plot = example_action
train = splits[action_to_plot]["train"]
test = splits[action_to_plot]["test"]

X_train = np.arange(len(train)).reshape(-1, 1)
y_train = train['change_percent']
X_test = np.arange(len(train), len(train) + len(test)).reshape(-1, 1)
y_test = test['change_percent']

# entraine le modele pour l'action
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# erreurs
residuals = y_test - predictions

# Visualise erreurs
plt.figure(figsize=(10, 5))
plt.scatter(test['date'], residuals, color='purple', alpha=0.7)
plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.title(f"R�sidus pour {action_to_plot}")
plt.xlabel("Date")
plt.ylabel("R�sidus (erreurs)")
plt.show()

     












