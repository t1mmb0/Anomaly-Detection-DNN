# Anomaly-Detection-DNN

**Disclaimer**: Dieses Repository wurde im Rahmen einer Bachelorarbeit an der Universität Jena erstellt. Die Erstellung des Programmcodes wurde durch Copilot und generative KI unterstützt.

## Dateistruktur

Die Dateien zu beiden Modellen beginnen immer mit dem Modellnamen.  
Die Pipeline wird verwendet, um eine neue Instanz zu trainieren.  
Die `Structure` beinhaltet alle Methoden, die für die Konstruktion einer neuen Instanz benötigt werden.  
`main.py` – Hier werden die Modelle visualisiert.  
Vortrainierte Modellinstanzen liegen in `trained_models`.  
In der `Config` können Variationen vorgenommen werden, um den Trainingsprozess anzupassen.

## Format der Daten

Die Trainings- und Testdaten liegen in folgendem Format vor:

- **Produktkategorie:**
  - `image.png`
  - **Anomaliekategorie 1:**
    - `sub_image_1.png`
    - `sub_image_2.png`
  - **Anomaliekategorie 2:**
    - `sub_image_3.png`

### Erklärung der Datenstruktur:
- **Produktkategorie**: Dies ist der übergeordnete Ordner, der die Produktkategorie beschreibt.
  - **Anomaliekategorie**: Jede Anomaliekategorie enthält eine oder mehrere Bilder (z.B. `image.png`, `sub_image_1.png`).
  - **.png-Dateien**: Diese Bilddateien sind die Daten, die zur Modellierung und Analyse verwendet werden.
