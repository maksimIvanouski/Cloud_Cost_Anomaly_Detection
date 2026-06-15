# Plan prezentacji: Wykrywanie anomalii w kosztach chmury z wykorzystaniem ML

---

## Slajd 1 — Strona tytułowa

**Tytuł:** Wykrywanie anomalii w kosztach chmury z wykorzystaniem uczenia maszynowego

**Autorzy:** Student 1, Student 2, Student 3

**Przedmiot:** [nazwa przedmiotu]

**Data:** Czerwiec 2025

---

## Slajd 2 — Problem biznesowy

- Firmy korzystające z chmury (AWS, Azure, GCP) mogą doświadczyć nieoczekiwanych wzrostów kosztów
- Przyczyny: błędna konfiguracja, zapomniane środowiska testowe, skoki transferu, zduplikowane zasoby
- Brak wykrycia = straty finansowe (nawet tysiące dolarów dziennie)
- Ręczne monitorowanie nie skaluje się — potrzeba automatycznego systemu

**Wizualizacja:** Wykres kosztów w czasie z zaznaczonymi anomaliami (czerwone punkty)

---

## Slajd 3 — Zbiór danych

- Syntetyczny zbiór ~5000 rekordów kosztów chmurowych
- 3 dostawców × 4 regiony × 7 usług × 3 środowiska
- 8 typów anomalii, ~7% rekordów anomalnych
- Dlaczego dane syntetyczne:
  - Bezpieczeństwo (brak prywatnych danych rozliczeniowych)
  - Powtarzalność (RANDOM_STATE=67)
  - Pełna kontrola nad scenariuszami anomalii

**Wizualizacja:** Rozkład klas (Normal vs Anomaly) — wykres słupkowy

---

## Slajd 4 — Inżynieria cech

- **Cechy temporalne:** dzień tygodnia, weekend, miesiąc
- **Cechy trendowe:** średnia krocząca 3/7-dniowa, stosunek kosztu do średniej
- **Cechy efektywności:** koszt/zasób, koszt/godzinę obliczeniową, koszt/GB transferu
- **Wskaźniki pomocnicze:** drogi development, wysoki transfer, nietypowa aktywność weekendowa
- **Kluczowa zasada:** obliczenia wyłącznie z przeszłych wartości (brak wycieku danych)

**Wizualizacja:** Tabela z przykładami cech i ich wartościami

---

## Slajd 5 — Pipeline ML

**Diagram:**
Generowanie danych → Walidacja → Inżynieria cech → Przetwarzanie wstępne → Podział train/test → Modele → Ewaluacja → Predykcja

- `RobustScaler` — odporny na outlier'y (skoki kosztów)
- `OneHotEncoder` z `handle_unknown="ignore"` — obsługuje nowe kategorie
- Podział 80/20 ze stratyfikacją (zachowanie proporcji klas)
- Przetwarzanie wstępne dopasowane **wyłącznie** na danych treningowych

---

## Slajd 6 — Modele

| # | Model | Typ | Opis |
|---|---|---|---|
| 1 | Baseline | Punkt odniesienia | Zawsze przewiduje klasę większościową |
| 2 | Regresja Logistyczna | Nadzorowany | Model liniowy z `class_weight="balanced"` |
| 3 | Las Losowy | Nadzorowany | Zespół drzew z GridSearchCV (cv=4, scoring=f1) |
| 4 | Las Izolacyjny | Nienadzorowany | Izoluje obserwacje nietypowe bez etykiet |

**Kluczowa różnica:** modele nadzorowane uczą się z etykiet, las izolacyjny — nie

---

## Slajd 7 — Wyniki

**Tabela porównawcza:**

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Baseline | ~0.93 | 0.00 | 0.00 | 0.00 | 0.50 |
| Regresja Logistyczna | ~ | ~ | ~ | ~ | ~ |
| Las Losowy | ~ | ~ | ~ | ~ | ~ |
| Las Izolacyjny | ~ | ~ | ~ | ~ | ~ |

**Kluczowy wniosek:** Accuracy nie wystarczy!
- Baseline: ~93% accuracy, ale 0% recall — nie wykrywa żadnej anomalii
- Wysoka accuracy ≠ skuteczne wykrywanie anomalii
- Właściwe metryki: F1-score, recall, precision

---

## Slajd 8 — Wizualizacja wyników

- **Macierz pomyłek** najlepszego modelu
  - Fałszywy alarm (FP) = badanie normalnego kosztu → strata czasu
  - Przeoczona anomalia (FN) = firma traci pieniądze → straty finansowe
  - **Który błąd groźniejszy?** FN — przeoczenie anomalii!

- **Krzywa ROC** — porównanie wszystkich modeli na jednym wykresie

**Wizualizacja:** Macierz pomyłek (heatmapa) + krzywa ROC (wielomodelowa)

---

## Slajd 9 — Interpretowalność

**Top cechy (Las Losowy):**

1. `cost_to_rolling_7_day_ratio` — stosunek kosztu do średniej 7-dniowej
2. `cost_usd` — wartość kosztu
3. `cost_to_rolling_3_day_ratio` — stosunek kosztu do średniej 3-dniowej
4. `compute_hours` — godziny obliczeniowe
5. `data_transfer_gb` — transfer danych

**Wniosek:** Anomalie wykrywane przez **KONTEKST** (odchylenie od wzorca), nie tylko wartość kosztu

**Przykład:**
- dev EC2 za 1200 USD przy średniej 100 USD → stosunek 12x → **ANOMALIA**
- prod EC2 za 420 USD przy średniej 405 USD → stosunek ~1x → **NORMALNY**

**Wizualizacja:** Wykres ważności cech (horizontal bar chart)

---

## Slajd 10 — Demo

**Uruchomienie `src/predict.py` na żywo:**

**Przykład 1: Normalny koszt produkcyjny**
```
Provider: AWS | Service: EC2 | Env: production | Cost: $420.00
Rolling 7-day avg: $405.00
→ Prediction: NORMAL ✅ | Confidence: ~89%
```

**Przykład 2: Anomalia w development**
```
Provider: AWS | Service: EC2 | Env: development | Cost: $1,200.00
Rolling 7-day avg: $100.00
→ Prediction: ANOMALY 🚨 | Confidence: ~94%
Suspicious indicators: cost 12x above average, expensive dev, high compute
```

---

## Slajd 11 — Koncepcja architektury chmurowej

**Jak system mógłby działać w firmie:**

1. **Eksport danych** — dane rozliczeniowe z AWS/Azure/GCP (API)
2. **Object Storage** — przechowywanie surowych danych (S3/Blob/GCS)
3. **Scheduled Job** — cron/Cloud Function uruchamia pipeline
4. **ML Pipeline** — preprocessing → feature engineering → scoring
5. **Model Registry** — wersjonowanie modeli
6. **Dashboard** — Streamlit/Grafana — wizualizacja podejrzanych skoków
7. **Alerting** — email/Slack do zespołu FinOps/DevOps

*Projekt uniwersytecki działa lokalnie — powyższe to koncepcja architektury docelowej*

**Wizualizacja:** Diagram architektury (blokowy)

---

## Slajd 12 — Wnioski

- ✅ ML skutecznie wspiera wykrywanie anomalii kosztowych w chmurze
- ✅ Las Losowy — najlepszy balans między precyzją a czułością
- ✅ Kluczowe: cechy kontekstowe (trendy, środowisko), nie sam koszt
- ✅ Accuracy nie wystarczy przy nierównowadze klas — pułapka dokładności
- ✅ Projekt łączy Data Science i Cloud Engineering

**Możliwe rozszerzenia:**
- Integracja z real-time billing API
- Automatyczne alerty (email/Slack)
- Modele szeregów czasowych (Prophet, LSTM)
- Klasyfikacja wieloklasowa typów anomalii
- MLflow do śledzenia eksperymentów

**Dziękujemy za uwagę!**
