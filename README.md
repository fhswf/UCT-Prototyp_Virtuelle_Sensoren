# Temperatur Interpolation mit virtuellen Sensoren - Urban Climate Twin Soest

Dieses Projekt interpoliert die Temperaturkorrektur durch virtuelle Sensoren auf Rasterzellen
und schreibt die Vorhersagen als CSV- und Shapefile-Dateien. Dazu werden
Messungen und Rasterinformationen aus einer PostgreSQL/PostGIS-Datenbank
geladen, aufbereitet und mit einem trainierten Keras-Modell ausgewertet.

## Urban Climate Twin Soest

 [Urban Climate Twin (UCT) Soest](https://www.fh-swf.de/de/forschung___transfer_4/forschungsprojekte_1/forschungsprojekt_uct.php) ist ein digitaler Zwilling des Soester Stadtklimas. Er führt Messdaten, Geodaten und weitere Klimadaten zusammen und macht das Stadklima räumlich nachvollziehbar. Bürger und die Stadtverwaltung können so die Auswirkungen von Klimaveränderungen auf das Stadtgebiet besser verstehen und Maßnahmen zur Anpassung an den Klimawandel entwickeln.


## Ablauf

`main.py` führt den gesamten Verarbeitungslauf aus:

1. Lädt Messungen, Quellen und Rasterdaten aus der konfigurierten Datenbank.
2. Bereitet Messwerte und Merkmale auf. Je nach Konfiguration werden
   SenseBox-Nachbarn, DWD-Werte, Rastermerkmale und Zeitmerkmale einbezogen.
3. Ordnet Messwerte den Rasterzellen und den konfigurierten Zeitpunkten zu.
   Bei längeren Zeiträumen werden stündliche Mediane gebildet.
4. Füllt fehlende Werte gemäß `transfer_values_method` auf und skaliert die
   Eingabemerkmale.
5. Berechnet mit dem Modell die Stage-2-Vorhersagen und exportiert die
   Ergebnisse.

Das Modell erwartet die in `preprocessing/utils.py` definierte
Merkmalsreihenfolge. Änderungen an den Eingabemerkmalen müssen daher mit dem
Training des Modells abgestimmt werden.

## Voraussetzungen

- Python mit kompatiblen Versionen der verwendeten Bibliotheken
- PostgreSQL-Datenbank mit PostGIS und den in der Konfiguration verwendeten
  Tabellen und Spalten
- gültige, zum Modell passende Gewichts- und Scaler-Dateien
- Zugangsdaten zur Datenbank


## Konfiguration

### Datenbankzugang

Lege im Projektverzeichnis eine `.env`-Datei an. Sie wird von `main.py` über
`python-dotenv` eingelesen:

```dotenv
DB_USER=<datenbank-benutzer>
DB_PASSWORD=<datenbank-passwort>
DB_HOST=<datenbank-host>
DB_PORT=<datenbank-port>
```

### Lauf und Modell

Passe `assets/config.yaml` vor dem Start an:

- `stage2_specifications.start` und `end`: Zeitraum der Verarbeitung; wenn keine Werte eingegeben werden wird die aktuelle UTC-Zeit minus 4 Stunden als Start und minus 3 Stunden + 1 Sekunde als Ende verwendet
- `stage2_specifications.file_paths.model` und `scaler`: Pfade zu den
  zusammengehörenden Modellgewichten und Scaler-Dateien
- `stage2_specifications.transfer_values_method`: Verfahren zur Auffüllung
  fehlender Werte, zum Beispiel `median_all`
- `stage2_specifications.cells_to_predict`: `null` für alle Rasterzellen oder
  eine Liste von Raster-IDs
- `preprocessing`: Auswahl und Verhalten der Datenaufbereitung
- `kriging_benchmark.enabled`: optionaler Kriging-Vergleich

Die in der Konfiguration eingetragenen Modell- und Scaler-Pfade müssen
tatsächlich existieren. Prüfe sie insbesondere nach dem Wechsel oder Austausch
der Artefakte; Gewichte und Scaler müssen zum selben Modell und zur erwarteten
Merkmalsreihenfolge passen.

## Ergebnisse

Die endgültigen Dateien erhalten Namen nach dem Muster
`stage2_predictions_<start>-<ende>_<methode>` und werden im aktuellen
Arbeitsverzeichnis gespeichert:

- `.csv` mit den Vorhersagewerten und zugehörigen Daten
- `.shp` mit den räumlichen Vorhersagen (zusammen mit den zugehörigen
  Shapefile-Dateien wie `.dbf`, `.shx` und `.prj`)

## Lizenz

Copyright 2026 Urban Climate Twin, Fachhochschule Südwestfalen

Lizenziert unter der [Apache License 2.0](LICENSE.md).
