# Rozpoznawanie tekstu oraz emocji z nagrań

## Projekt w ramach przedmiotu Programowanie w języku Python.
Za pomocą torchaudio i pytorch od zera (dysponując wyłącznie zbiorem danych) chcę zaprojektować, 
utworzyć i wytrenować model który pozwoli na rozpoznanie dominującej w wypowiedzi emocji.

### Analiza MoSCoW:

#### Must have:
- [x] Dataset, skorzystano z RAVDESS[^1]
- [x] Model tworzony od zera i ćwiczony od zera
- [x] Opcja odtworzenia użytkownikowi analizowanego nagrania i wypisanie mu zaklasyfikowanej emocji

#### Should have
- [x] jakieś proste (niekoniecznie ładne) GUI do komunikacji z użytkownikiem
- [ ] Opcja zapodania dowolnego nagrania (jeszcze nie wiem w jakim formacie) w języku angielskim

#### Could have
- [ ] jakieś proste i, w miarę możliwości, "przyjemne wizualnie" GUI
- [ ] Nagranie na żywo z wykorzystaniem mikrofonu i klasyfikacja na żywo (język angielski)

[^1]: This whole project wouldn't be possible without the amazing RAVDESS dataset: https://zenodo.org/record/1188976
