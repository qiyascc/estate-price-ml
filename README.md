# estate-price-ml

Azərbaycan əmlak bazarı üçün qiymət təxmini modelləri. Rayon, otaq sayı, sahə,
mərtəbə, metro və digər xüsusiyyətlərə əsasən əmlakın qiymətini proqnozlaşdırır.
Modellər Estate Scraper topladığı canlı bazadakı 872 min real elan üzərində
öyrədilib və hazır şəkildə `models/` qovluğunda yerləşir.

## Hazır modellər

| Fayl | Nə üçün | Dəqiqlik (R2) | Orta xəta (MAPE) |
|------|---------|---------------|-------------------|
| `models/satis.joblib` | Satış qiyməti | 0.82 | 18.5 faiz |
| `models/vasitechi_satis.joblib` | Vasitəçi satışı | 0.84 | 16.6 faiz |
| `models/sahibkar_satis.joblib` | Sahibkar satışı | 0.68 | 27.6 faiz |
| `models/kiraye.joblib` | Aylıq kirayə | 0.63 | 37 faiz |

Bütün metrikalar `models/hesabat.json` faylındadır (bölgə üzrə qiymətlər,
sahibkar ilə vasitəçi fərqi, kirayə gəlirliliyi də orada).

## Quraşdırma

```
git clone https://github.com/qiyascc/estate-price-ml.git
cd estate-price-ml
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Sorğu, qiymət təxmini

Ən sadə yol `estimate` funksiyasıdır. Əmlakın xüsusiyyətlərini verirsən, o da
qiyməti qaytarır. Qiymət vermək lazım deyil, məhz onu proqnozlaşdırır.

```python
from estate_price_ml.estimate import estimate

qiymet = estimate("models/kiraye.joblib", {
    "deal_type": "rent",
    "district": "Nərimanov",
    "metro": "Nəriman Nərimanov",
    "rooms": 2,
    "area": 70,
    "floor_current": 5,
    "floor_total": 9,
    "property_type": "Mənzil / Yeni tikili",
})
print(qiymet)
```

Nəticə:

```
[847.88]
```

Yəni Nərimanovda 2 otaqlı 70 kvadratmetr mənzilin təxmini aylıq kirayəsi
təxminən 848 AZN. Satış qiyməti üçün satış modelini işlət:

```python
from estate_price_ml.estimate import estimate

qiymet = estimate("models/satis.joblib", {
    "deal_type": "sale",
    "district": "Yasamal",
    "rooms": 3,
    "area": 100,
    "floor_current": 7,
    "floor_total": 12,
    "property_type": "Mənzil / Yeni tikili",
})
print(qiymet)
```

Nəticə:

```
[241863.95]
```

## Bir neçə əmlakı birdən

`estimate` funksiyasına siyahı da vermək olar, hər biri üçün qiymət qaytarır:

```python
from estate_price_ml.estimate import estimate

qiymetler = estimate("models/kiraye.joblib", [
    {"district": "Nərimanov", "rooms": 1, "area": 45},
    {"district": "Nərimanov", "rooms": 2, "area": 70},
    {"district": "Yasamal", "rooms": 3, "area": 100},
])
print(qiymetler)
```

## Əmr sətrindən sorğu

Tək əmlak üçün terminaldan da sorğu vermək olar:

```
estate-price-ml estimate --model models/kiraye.joblib \
  --deal rent --district "Nərimanov" --metro "Nəriman Nərimanov" \
  --rooms 2 --area 70 --floor 5 --floors 9
```

Çıxış:

```
848
```

## Sorğuda verilə bilən sahələr

Nə qədər çox məlumat versən, təxmin bir o qədər dəqiqdir. Heç biri məcburi
deyil, verilməyənlər üçün boş dəyər götürülür.

| Sahə | İzah |
|------|------|
| `deal_type` | sale və ya rent |
| `city` | şəhər |
| `district` | rayon, məsələn Nərimanov |
| `neighborhood` | qəsəbə |
| `metro` | ən yaxın metro |
| `property_type` | əmlak növü |
| `rooms` | otaq sayı |
| `area` | sahə kvadratmetrlə |
| `floor_current` | mərtəbə |
| `floor_total` | binanın ümumi mərtəbəsi |
| `latitude`, `longitude` | koordinatlar |
| `is_daily_rent` | günlük kirayədirsə 1 |

Rayon və metro adları bazadakı yazılışla üst üstə düşməlidir, məsələn
`Nərimanov`, `Nəriman Nərimanov`. Yanlış yazılış səhv saymaz, sadəcə o
xüsusiyyət nəzərə alınmaz və model qalan məlumatlarla təxmin edər.

## Bazar analizi

Rayon üzrə orta qiymətlər, sahibkar ilə vasitəçi fərqi və kirayə gəlirliliyi
üçün:

```
estate-price-ml analyze --db /yol/db.sqlite3 --region district
```

## Modeli yenidən öyrətmək

Yeni data ilə modelləri yenidən öyrətmək üçün baza yolunu ver:

```
estate-price-ml train --db /yol/db.sqlite3 --out models
```

Baza yolu `ESTATEPRICEML_DB` mühit dəyişəni ilə də verilə bilər. Öyrətmə
nəticəsində `models/` qovluğunda dörd model faylı və `hesabat.json` yaranır.
Baza repozitoriyaya daxil edilmir, yalnız öyrədilmiş modellər saxlanılır.

## Necə işləyir

Model HistGradientBoosting alqoritmidir, qiymətin loqarifmi üzərində öyrədilir.
Xüsusiyyətlər: sahə, otaq sayı, otaq başına sahə, mərtəbə və mərtəbə nisbəti,
koordinatlar və şəhər mərkəzinə məsafə, metro, rayon və qəsəbə, əmlak növü,
başlıqdan çıxarılan açar sözlər. Bütün qiymətlər AZN valyutasına çevrilir.
Satış və kirayə üçün ayrıca, sahibkar və vasitəçi üçün də ayrıca modellər var.

## Testlər

```
pip install pytest
pytest
```
