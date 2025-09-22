# Projets-de-classe-M2

## Prérequis

Avant de lancer l'application, assurez-vous d'avoir installé les bibliothèques Python nécessaires. Vous pouvez les installer via pip :

"pip install gradio opencv-python numpy Pillow fpdf"


## Comment Utiliser

1.  **Sauvegardez le code** dans un fichier nommé `nomdevotrefichier.py`.
2.  **Placez le logo** (optionnel) dans le même répertoire`logo.png` .
3.  **Lancez l'application** depuis votre terminal :

    ```bash
    python nomdevotrefichier.py
    ```

4.  **Ouvrez votre navigateur** et allez à l'URL affichée dans le terminal (Local `http://127.0.0.1:7860`) sinon modifiez share=True pour avoir une URL Web de la
   forme "https://xxxxxxxxxxxxxx.gradio.live" 



### Étapes dans l'interface web
1.  **Charger une image** : Utilisez le bloc "🖼️ 1. Entrée & Actions" pour glisser-déposer une image ou cliquer pour en sélectionner une. L'image apparaîtra également dans la zone de prévisualisation.
2.  **Analyser** : Cliquez sur le bouton "🔬 Analyser l'image". Les résultats de l'analyse s'afficheront dans l'accordéon "🔍 Voir les résultats de l'analyse" en bas de la page, répartis dans différents onglets (Infos & Stats, Canaux, Histogramme, Filtres).
3.  **Utiliser les actions interactives** (optionnel) :
    - **Lecture de Pixel** : Entrez des coordonnées X/Y et cliquez sur "📍 Lire Pixel".
    - **Recadrage** : Définissez les coordonnées et dimensions de la zone à recadrer, puis cliquez sur "✂️ Recadrer". L'image dans la zone de prévisualisation sera mise à jour avec l'image recadrée.
4.  **Ajouter au rapport** : Si l'analyse vous convient, cliquez sur "➕ Ajouter au rapport". L'ID de l'analyse apparaîtra dans la liste déroulante "Analyses dans le rapport".
5.  **Générer un PDF** :
    - Pour un rapport sur une analyse spécifique : sélectionnez-la dans la liste déroulante et cliquez sur "Télécharger le rapport sélectionné".
    - Pour un rapport consolidant toutes les analyses ajoutées : cliquez sur "🌍 Générer le Rapport Global".
6.  **Télécharger le fichier** : Le lien de téléchargement du PDF apparaîtra dans le bloc "📥 Fichier PDF".
