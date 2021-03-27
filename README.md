## Pré-requis

Il faut avoir les fichiers abundance.txt et marker_presence.txt présent dans le même répertoire que les scripts.

## Split des données

On commence par éliminer les doublons (échantillons du provenant du même individu) et séparer les données en training est testing set. On utilisera seulement le testing set à la fin pour évaluer les modèles optimisés avec le training set.

Pour faire cette première étape, on n'a qu'a exécuter le script data.py :

```
python data.py
```

Le script va créer 2 fichiers : train_idx.npy et test_idx.npy. C'est deux fichiers contiennent les indexes des échantillons de chaque dataset. On pourra utiliser ces fichiers pour charger le dataset voulu. (C'est déjà fait. On peut utiliser les fichiers déjà présents.)

## Chargement des données

Il suffit d'utiliser la fonction load_x_and_y contenu dans data.py. La fonction prend en paramètre le chemin du fichier des indexes à charger. Un exemple :

```python
#On importe la fonction
from data import load_x_and_y

xTest, yTest = load_x_and_y( 'test_idx.npy' )
print( 'Testing set shapes', xTest.shape, yTest.shape)

xTrain, yTrain = load_x_and_y( 'train_idx.npy' )
print( 'Training set shapes', xTrain.shape, yTrain.shape)
```
