# Anomaly-Detection-DNN
 
Disclamer: Dieses Repository wurde im Rahmen einer Bachelorarbeit an der Universität Jena erstellt.

## Dateistruktur

Die Files zu beiden Modellen beginnen immer mit dem Modellnamen.
Die Pipeline wird verwendet, um eine neue Instanz zu trainieren.
Die Structure beinhaltet alle Methoden, die für die Konstruktion einer neuen Instanz benötigt werden.
main.py - hier werden die Modelle visualisiert.
vortrainierte Modellinstanzen liegen in trained_models
in der Config können Variationen vorgenommen werden, um den Trainingsprozess anzupassen.

## Format der Daten

Die Trainings- und Testdaten liegen in folgendem Format vor:

- **Produktkategorie:**
  - `image.png`
  - **Anomaliekategorie 1:**
    - `sub_image_1.png`
    - `sub_image_2.png`
  - **Anomaliekategorie 2:**
    - `sub_image_3.png`
