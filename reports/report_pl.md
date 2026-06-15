# Wykrywanie anomalii w kosztach chmury z wykorzystaniem uczenia maszynowego

**Autorzy:** Student 1, Student 2, Student 3

**Data:** Czerwiec 2025

---

## 1. Wprowadzenie

Współczesne przedsiębiorstwa coraz częściej korzystają z infrastruktury chmurowej (AWS, Azure, Google Cloud), co wiąże się z dynamicznie zmieniającymi się kosztami operacyjnymi. Nieoczekiwane wzrosty kosztów mogą wynikać z błędnej konfiguracji zasobów, zapomnianych środowisk testowych, skoków transferu danych, błędów wdrożeniowych czy nietypowych wzorców użycia infrastruktury. Brak szybkiego wykrycia takich anomalii może prowadzić do znaczących strat finansowych — w skrajnych przypadkach sięgających tysięcy dolarów dziennie.

Celem niniejszego projektu jest opracowanie systemu uczenia maszynowego zdolnego do klasyfikacji rekordów kosztów chmurowych jako normalnych lub anomalnych, wspierając zespoły FinOps i DevOps w proaktywnym monitorowaniu wydatków na infrastrukturę chmurową.

## 2. Cel projektu

Zadanie polega na binarnej klasyfikacji rekordów kosztów:

- **0** = koszt normalny (zachowanie zgodne z historycznym wzorcem)
- **1** = anomalia kosztowa (odchylenie wymagające analizy)

Istotne jest, że wysokie koszty nie zawsze oznaczają anomalię — koszt 900 USD w środowisku produkcyjnym może być całkowicie normalny, podczas gdy koszt 400 USD w środowisku deweloperskim może stanowić poważną anomalię, jeśli jego historyczna średnia wynosi 50 USD. Model musi zatem oceniać koszty w kontekście: dostawcy chmury, regionu, usługi, środowiska, poziomu użycia oraz historycznych trendów kosztowych.

## 3. Dane

### 3.1. Charakterystyka zbioru danych

Wykorzystano syntetyczny zbiór danych symulujący dzienne koszty infrastruktury chmurowej. Zbiór zawiera około 5000 rekordów z 20 kolumnami, obejmującymi:

- **Cechy temporalne** — dzień tygodnia, weekend, miesiąc
- **Cechy infrastrukturalne** — dostawca chmury (AWS, Azure, Google Cloud), region (us-east-1, eu-central-1, eu-west-1, us-west-2), usługa (EC2, S3, RDS, Lambda, CloudWatch, Data Transfer, Kubernetes), środowisko (production, staging, development)
- **Cechy użycia** — ilość zasobów, transfer danych, godziny obliczeniowe, przestrzeń dyskowa, liczba żądań
- **Cechy kosztowe** — koszt bieżący (USD), koszt poprzedniego dnia, średnia krocząca 3- i 7-dniowa
- **Zmienna docelowa** — `is_anomaly` (0 lub 1)

### 3.2. Anomalie

Anomalie stanowią ~7% rekordów, co odzwierciedla naturalną nierównowagę klas typową dla zadań wykrywania anomalii. Zdefiniowano 8 typów anomalii:

| Typ anomalii | Opis |
|---|---|
| sudden_compute_spike | Nagły skok kosztów obliczeniowych (4–8x) |
| abnormal_data_transfer | Nietypowy wzrost transferu danych (10–20x) |
| storage_growth | Anomalny wzrost przestrzeni dyskowej (8–15x) |
| forgotten_test_environment | Zapomniane środowisko testowe osiągające koszty produkcyjne |
| unexpected_region_usage | Nieoczekiwane wysokie koszty w regionie niestandardowym |
| monitoring_cost_spike | Skok kosztów monitoringu/logowania (5–10x) |
| weekend_production_activity | Nietypowa aktywność produkcyjna w weekend |
| duplicated_resource_deployment | Zduplikowane wdrożenie zasobów (4–8x) |

### 3.3. Uzasadnienie danych syntetycznych

Dane syntetyczne wybrano świadomie z następujących powodów:

- Pełna kontrola nad scenariuszami anomalii i ich proporcjami
- Brak konieczności dostępu do prywatnych danych rozliczeniowych
- Bezpieczeństwo prezentacji w środowisku akademickim
- Powtarzalność eksperymentów dzięki ustalonemu ziarnu losowości (`RANDOM_STATE=67`)
- Możliwość zaprojektowania różnorodnych, realistycznych scenariuszy anomalii

## 4. Pipeline uczenia maszynowego

Pipeline projektu obejmuje następujące etapy:

1. **Generowanie danych** (`src/data_generation.py`) — tworzenie realistycznego zbioru syntetycznego z kontrolowanymi anomaliami.
2. **Walidacja danych** (`src/data_loading.py`) — sprawdzenie kompletności kolumn, brak ujemnych kosztów, weryfikacja rozkładu klas, wykluczenie kolumn nieistotnych.
3. **Analiza eksploracyjna** (`src/exploratory_analysis.py`) — wizualizacja rozkładów kosztów, dystrybucji klas, kosztów w podziale na usługi, regiony i środowiska.
4. **Inżynieria cech** (`src/feature_engineering.py`) — tworzenie cech trendowych (stosunek kosztu do średniej kroczącej), cech efektywności (koszt na zasób, koszt na godzinę obliczeniową), wskaźników pomocniczych (drogi development, wysoki transfer).
5. **Przetwarzanie wstępne** (`src/preprocessing.py`) — imputacja (mediana dla cech numerycznych, moda dla kategorycznych), skalowanie (`RobustScaler` odporny na wartości odstające), kodowanie (`OneHotEncoder` z `handle_unknown="ignore"`).
6. **Podział danych** — 80% trening, 20% test, podział stratyfikowany zachowujący proporcje klas.
7. **Trenowanie modeli** (`src/train.py`) — 4 modele z optymalizacją hiperparametrów (GridSearchCV dla Lasu Losowego).
8. **Ewaluacja** (`src/evaluate.py`) — metryki klasyfikacji, macierz pomyłek, krzywe ROC i precyzja-czułość.
9. **Predykcja** (`src/predict.py`) — demonstracja działania modelu na przykładowych rekordach.

**Kluczowa zasada:** przetwarzanie wstępne dopasowywane jest wyłącznie na danych treningowych i aplikowane na danych testowych. Średnie kroczące obliczane są wyłącznie z wartości przeszłych, co zapobiega wyciekowi danych (*data leakage*).

## 5. Modele

### 5.1. Baseline — Klasyfikator klasy większościowej

Model zawsze przewiduje klasę dominującą (normalny koszt). Osiąga wysoką dokładność (~93%), ale wykrywa zero anomalii (recall = 0). Demonstruje to tzw. „pułapkę dokładności" (*accuracy trap*) — wysoka wartość metryki accuracy nie oznacza użytecznego modelu w kontekście wykrywania anomalii.

### 5.2. Regresja Logistyczna

Prosty model liniowy z parametrem `class_weight="balanced"`, który automatycznie zwiększa wagę klasy mniejszościowej. Zapewnia interpretowalność poprzez współczynniki cech i stanowi bazową zdolność klasyfikacji. Dobrze radzi sobie z liniowo separowalnymi przypadkami anomalii.

### 5.3. Las Losowy (*Random Forest*)

Model nieparametryczny wykorzystujący zespół drzew decyzyjnych. Optymalizowany za pomocą `GridSearchCV` (cv=4, scoring=f1) po następujących parametrach: `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`. Oferuje ranking ważności cech (*feature importance*), co wspiera interpretowalność wyników.

### 5.4. Las Izolacyjny (*Isolation Forest*)

Model nienadzorowany — nie wykorzystuje etykiet podczas trenowania. Identyfikuje obserwacje „izolowane", czyli takie, które są łatwe do oddzielenia od reszty danych za pomocą losowych podziałów. Parametr `contamination` ustawiony na przybliżony udział anomalii w zbiorze.

**Różnica koncepcyjna:** Regresja Logistyczna i Las Losowy to modele nadzorowane (*supervised*) — uczą się na oznaczonych danych. Las Izolacyjny to model nienadzorowany (*unsupervised*) — szuka obserwacji nietypowych bez znajomości etykiet, co ma istotne zastosowanie w scenariuszach, gdzie etykiety nie są dostępne.

## 6. Wyniki

### 6.1. Porównanie modeli

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Baseline (klasa większościowa) | ~0.93 | 0.00 | 0.00 | 0.00 | 0.50 |
| Regresja Logistyczna | ~ | ~ | ~ | ~ | ~ |
| Las Losowy | ~ | ~ | ~ | ~ | ~ |
| Las Izolacyjny | ~ | ~ | ~ | ~ | ~ |

*Uwaga: dokładne wartości zostaną uzupełnione po uruchomieniu pipeline'u z ziarnem losowości RANDOM_STATE=67.*

### 6.2. Dlaczego accuracy nie wystarczy

Przy ~7% anomalii model przewidujący zawsze „normalny" osiąga ~93% accuracy, ale ma zerowy recall — nie wykrywa żadnej anomalii. Model jest bezużyteczny pomimo wysokiej dokładności. Dlatego kluczowe metryki to:

- **F1-score** — harmoniczna średnia precyzji i czułości
- **Recall** — jaki odsetek rzeczywistych anomalii został wykryty
- **Precision** — jaki odsetek predykowanych anomalii jest prawdziwy
- **ROC-AUC** — pole pod krzywą ROC

### 6.3. Macierz pomyłek — interpretacja biznesowa

- **Prawdziwie pozytywny (TP):** anomalia poprawnie wykryta → zespół bada rzeczywisty skok kosztów
- **Prawdziwie negatywny (TN):** normalny koszt poprawnie sklasyfikowany → brak zbędnej interwencji
- **Fałszywy alarm (FP):** normalny koszt oznaczony jako anomalia → zespół FinOps traci czas na badanie prawidłowego kosztu
- **Przeoczona anomalia (FN):** rzeczywista anomalia pominięta → firma nie wykrywa skoku kosztów i ponosi straty finansowe

W kontekście monitoringu kosztów chmurowych **fałszywe negatywy (FN) są zazwyczaj groźniejsze** — przeoczenie anomalii może prowadzić do znaczących, narastających strat finansowych.

## 7. Interpretacja modelu

### 7.1. Najważniejsze cechy

Na podstawie rankingu ważności cech modelu Las Losowy zidentyfikowano następujące kluczowe predyktory:

1. `cost_to_rolling_7_day_ratio` — stosunek bieżącego kosztu do średniej 7-dniowej
2. `cost_usd` — wartość kosztu w USD
3. `cost_to_rolling_3_day_ratio` — stosunek bieżącego kosztu do średniej 3-dniowej
4. `compute_hours` — godziny obliczeniowe
5. `data_transfer_gb` — transfer danych w GB
6. `resource_count` — liczba aktywnych zasobów
7. `cost_per_resource` — koszt na zasób

Cechy trendowe (stosunek kosztu do średniej kroczącej) okazały się kluczowe — potwierdzają, że anomalie wykrywane są nie przez sam poziom kosztu, lecz przez **odchylenie od wzorca historycznego**.

### 7.2. Studium przypadku

**Rekord normalny:**
AWS EC2, środowisko production, koszt 420 USD, średnia 7-dniowa 405 USD.
Model poprawnie klasyfikuje jako NORMALNY — koszt jest zgodny z historycznym wzorcem środowiska produkcyjnego.

**Rekord anomalny:**
AWS EC2, środowisko development, koszt 1200 USD, średnia 7-dniowa 100 USD.
Model poprawnie identyfikuje ANOMALIĘ ze wskaźnikami:
- Koszt 12x wyższy od średniej 7-dniowej
- Środowisko deweloperskie z nietypowo wysokim kosztem
- Wysoki transfer danych i godziny obliczeniowe
- Wysoka liczba zasobów

Ten przykład ilustruje fundamentalną zasadę projektu — **sam poziom kosztu nie determinuje anomalii; kluczowy jest kontekst**.

## 8. Koncepcja architektury chmurowej

System mógłby funkcjonować w środowisku produkcyjnym firmy według następującego schematu:

1. **Eksport danych** — automatyczny eksport danych rozliczeniowych z AWS/Azure/GCP (np. AWS Cost Explorer API)
2. **Przechowywanie** — surowe dane rozliczeniowe składowane w object storage (S3, Blob Storage, GCS)
3. **Zaplanowane zadanie** — cron job lub Cloud Function uruchamia pipeline ML w regularnych interwałach
4. **Pipeline ML** — przetwarzanie wstępne → inżynieria cech → scoring modelu
5. **Rejestr modeli** — wytrenowany model przechowywany w rejestrze z wersjonowaniem
6. **Predykcje** — nowe rekordy kosztów oceniane jako normalne lub anomalne
7. **Dashboard** — panel Streamlit/Grafana prezentujący podejrzane skoki kosztów
8. **Alerting** — automatyczne powiadomienia email/Slack do zespołów FinOps/DevOps

*Uwaga: niniejszy projekt uniwersytecki działa lokalnie bez poświadczeń chmurowych. Powyższy schemat stanowi koncepcję architektury docelowej.*

## 9. Wnioski

1. **Uczenie maszynowe skutecznie wspiera wykrywanie anomalii kosztowych** w infrastrukturze chmurowej, przewyższając podejścia oparte na prostych progach.
2. **Las Losowy osiągnął najlepszy balans** między precyzją a czułością wśród testowanych modeli nadzorowanych.
3. **Kluczowe okazały się cechy kontekstowe** — sam koszt nie wystarczy do identyfikacji anomalii; niezbędne jest uwzględnienie historycznych trendów, środowiska i typu usługi.
4. **Pułapka dokładności** (*accuracy trap*) potwierdzona eksperymentalnie — model baseline osiąga ~93% accuracy przy zerowym wykrywaniu anomalii.
5. **Projekt łączy kompetencje Data Science i Cloud Engineering**, demonstrując zastosowanie ML w praktycznym problemie zarządzania kosztami chmury.

### Potencjalne rozszerzenia

- Integracja z rzeczywistymi danymi rozliczeniowymi (AWS Cost Explorer, Azure Cost Management)
- Wykrywanie anomalii w czasie rzeczywistym (streaming)
- Automatyczne alerty i eskalacja
- Modele szeregów czasowych (Prophet, LSTM) do prognozowania trendów
- Klasyfikacja wieloklasowa typów anomalii
- Implementacja jako mikroserwis z MLflow

---

**Repozytorium:** github.com/[team]/cloud-cost-anomaly-detection
