# Fiducial Data Collector - Colectarea Sistematică de Date

## Prezentare Generală

Acest proiect se concentrează **exclusiv pe colectarea datelor** pentru markerii fiduciali folosind OAK-D Lite. Sistemul ghidează utilizatorul pas-cu-pas prin toate măsurătorile necesare.

## Inspirație din Cercetarea CopperTag

Bazat pe metodologia din articolul CopperTag, dar adaptat pentru:
- ✅ **Măsurători reale** (nu simulare)
- ✅ **Camera stereo-depth** (OAK-D Lite)
- ✅ **Ghidare interactivă** pentru utilizator
- ✅ **Combinații optimizate** pentru o singură persoană

## Combinații de Testare Optimizate

### **Markeri Selectați (7 tipuri reprezentative)**
1. **ArUco 4x4_50** - Standard industrial
2. **ArUco 6x6_250** - Echilibru precizie/viteză
3. **AprilTag 36h11** - Cel mai robust AprilTag
4. **QR Code** - Standard comercial
5. **RuneTag** - Reprezentant circular
6. **ChromaTag** - Reprezentant color-based
7. **CopperTag** - Reprezentant industrial robust

### **Condiții de Testare (Inspirate din CopperTag)**

#### **Test Set 1: Distanță (8 măsurători × 7 markeri = 56 teste)**
- 0.3m, 0.6m, 1.0m, 1.5m, 2.0m, 2.5m, 3.0m, 3.5m
- **Timp estimat**: ~2 ore

#### **Test Set 2: Rotație X (7 măsurători × 7 markeri = 49 teste)**
- -60°, -40°, -20°, 0°, 20°, 40°, 60°
- **Timp estimat**: ~1.5 ore

#### **Test Set 3: Rotație Y (7 măsurători × 7 markeri = 49 teste)**
- -60°, -40°, -20°, 0°, 20°, 40°, 60°
- **Timp estimat**: ~1.5 ore

#### **Test Set 4: Rotație Z (5 măsurători × 7 markeri = 35 teste)**
- 0°, 45°, 90°, 180°, 270°
- **Timp estimat**: ~1 oră

#### **Test Set 5: Ocluziune (4 măsurători × 7 markeri = 28 teste)**
- 5%, 10%, 15%, 20% (cu obiecte fizice)
- **Timp estimat**: ~1 oră

#### **Test Set 6: Iluminare (4 măsurători × 7 markeri = 28 teste)**
- Bright, Normal, Dim, Shadow
- **Timp estimat**: ~1 oră

**TOTAL: 245 teste în ~7 ore de colectare**

## Structura Proiectului

```
fiducial_data_collector/
├── README.md                    # Acest fișier
├── main_collector.py            # Script principal cu ghidare
├── config/
│   ├── test_configurations.py   # Configurațiile de testare
│   ├── marker_definitions.py    # Definițiile markerilor
│   └── measurement_protocol.py  # Protocolul de măsurare
├── detectors/
│   ├── opencv_detectors.py      # Detectori OpenCV (ArUco, AprilTag, QR)
│   ├── external_detectors.py    # Detectori externi (RuneTag, ChromaTag, etc.)
│   └── detector_manager.py      # Manager pentru toți detectorii
├── data_collection/
│   ├── oak_interface.py         # Interfața cu OAK-D Lite
│   ├── metrics_collector.py     # Colectarea metricilor
│   ├── user_guidance.py         # Ghidarea utilizatorului
│   └── data_saver.py           # Salvarea datelor
├── utils/
│   ├── system_monitor.py        # Monitorizare CPU/RAM
│   ├── progress_tracker.py      # Tracking progres
│   └── validation_helpers.py    # Validare date
├── markers/                     # Markerii pentru printare
│   ├── aruco/
│   ├── apriltag/
│   ├── qr/
│   └── custom/
└── datasets/                    # Datele colectate
    ├── raw_data/
    ├── processed/
    └── reports/
```

## Fluxul de Colectare

### **Pas 1: Pregătirea**
```
🖨️  Printează markerii din markers/
📏 Pregătește rigla pentru măsurarea distanțelor
💡 Pregătește surse de lumină pentru teste iluminare
📦 Pregătește obiecte pentru teste ocluziune
```

### **Pas 2: Calibrarea**
```
📷 Conectează OAK-D Lite
🎯 Calibrează camera automat
📐 Setează sistemul de coordonate
```

### **Pas 3: Colectarea Ghidată**
```
👤 Sistemul îți spune exact ce să faci:
   "Poziționează markerul ArUco 4x4_50 la 0.3m distanță"
   "Rotește camera cu 20° pe axa X"
   "Aplică umbră pe jumătate din marker"
   
📊 Colectează automat toate metricile
💾 Salvează datele în timp real
📈 Afișează progresul (Test 15/280)
```

## Ghidarea Interactivă

### **Exemplu de Interacțiune**
```
🎯 FIDUCIAL DATA COLLECTOR
📊 Progres general: 15/280 teste (5.4%)
⏱️  Timp rămas estimat: 7h 23min

📍 TEST CURENT: Distanță - ArUco 4x4_50
🎯 Instrucțiuni:
   1. Printează markerul ArUco 4x4_50 (5cm x 5cm)
   2. Lipește markerul pe o suprafață plană
   3. Poziționează markerul la EXACT 0.6m de cameră
   4. Asigură-te că markerul este perpendicular pe cameră
   5. Apasă ENTER când ești gata

📷 Camera detectează: ✅ Marker găsit
📏 Distanța măsurată: 0.58m (±2cm - OK)
⏱️  Colectare în curs... 10s

✅ Test completat!
📊 Rezultate:
   - Rata detecție: 98.5%
   - Timp procesare: 12.3ms
   - CPU: 45%, RAM: 1.2GB
   - Colțuri detectate: 4/4

➡️  Următorul test: ArUco 4x4_50 la 1.0m
```

## Metricile Colectate

### **Pentru Fiecare Test (10 metrici)**
1. **CPU utilizat** - % în timpul detecției
2. **Memorie consumată** - MB peak usage
3. **Timpul de procesare** - ms per frame
4. **Rata de detecție** - % frame-uri cu detecție
5. **Distanța măsurată** - vs distanța reală
6. **Precizia colțurilor** - eroarea în pixeli
7. **Stabilitatea ID** - consistența identificării
8. **Robustețea la mișcare** - detecție în timpul mișcării
9. **Calitatea depth** - validitatea datelor depth
10. **Scorul general** - metric agregat

## Avantajele Acestui Approach

### **Față de CopperTag**
- ✅ **Măsurători reale** vs simulare
- ✅ **Date depth** pentru poziționare 3D precisă
- ✅ **Ghidare pas-cu-pas** pentru reproductibilitate
- ✅ **Optimizat pentru o persoană** (8 ore vs săptămâni)

### **Față de Alte Studii**
- ✅ **Markeri diversi** (8 tipuri reprezentative)
- ✅ **Condiții realiste** (iluminare, ocluziune)
- ✅ **Metrici complete** (10 categorii)
- ✅ **Date structurate** pentru analiză ulterioară

## Rezultate Așteptate

### **Dataset Final**
- **245 teste** complete
- **~45GB date** (RGB + Depth + Metadata)
- **2450 metrici** individuale (10 × 245)
- **Raport automat** cu statistici

### **Aplicabilitate**
- **Cercetare academică** - dataset pentru publicații
- **Dezvoltare industrială** - alegerea markerilor optimi
- **Benchmarking** - comparația obiectivă a algoritmilor
- **Optimizare** - identificarea slăbiciunilor pentru îmbunătățiri

## Următorii Pași

1. **Implementarea scriptului principal** cu ghidare interactivă
2. **Integrarea detectorilor** pentru toți markerii
3. **Testarea cu OAK-D Lite** pentru validare
4. **Colectarea dataset-ului** în ~8 ore
5. **Generarea raportului** automat cu rezultate