Ziel
Stündliche Erfassung der Netherworld Monstercount Tabelle. Speicherung im Repo. Wöchentliche und monatliche Berichte. Discord Posts unter 2000 Zeichen.

Schnellstart
1. Repository anlegen und Dateien übernehmen.
2. In GitHub unter Settings und Secrets einen Secret mit Namen DISCORD_WEBHOOK hinzufügen. Inhalt ist die Discord Webhook URL.
3. Workflows aktivieren. Requirements sind in requirements.txt.

Was passiert
scrape-hourly läuft jede Stunde. Der Scraper lädt die Seite und schreibt
data raw Netherworld YYYY-MM-DD HH.json
data hourly Netherworld YYYY-MM-DD.json

generate-and-post-reports läuft stündlich. Das Skript prüft die lokale Europe Berlin Zeit. Bei Minuten 15 bis 19 laufen die Prüfungen.
Montag kurz nach Mitternacht entstehen Wochenberichte für die letzte Woche Montag bis Sonntag.
Am ersten Kalendertag kurz nach Mitternacht entstehen Monatsberichte für den letzten Monat.

Filter
Tage mit weniger als 1500 Gesamt Kills pro Spieler werden ignoriert.

Kategorien
inaktiv kleiner 500
normal 501 bis kleiner 3000
mittelaktiv 3000 bis 4000
hochaktiv 4000 oder mehr

Anpassungen
config config.yaml enthält Basis URL, Zeitzone, Schwellen und Limits.
