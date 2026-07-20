# estate-price-ml

Azərbaycan əmlak bazarı üçün qiymət proqnozu və bazar analizi kitabxanası.
Estate Scraper topladığı əmlak elanları üzərində işləyir. Modellər bölgə, otaq
sayı, sahə, mərtəbə, əmlak növü və satıcı tipi kimi xüsusiyyətlərdən istifadə
edərək qiyməti proqnozlaşdırır.

## Nə edir

Kitabxana əmlak bazarını dörd əsas seqmentdə analiz edir:

1. Satışda olan əmlakların analizi
2. Kirayədə olan əmlakların analizi
3. Sahibkar tərəfindən satılan əmlakların analizi
4. Vasitəçi tərəfindən satılan əmlakların analizi

Hər seqment üçün ayrıca model qurulur və bölgə üzrə qiymət proqnozu verilir.
Əlavə olaraq bazar hesabatları hazırlanır: bölgə üzrə medyan qiymət və kvadrat
metr qiyməti, sahibkar ilə vasitəçi qiymət fərqi, kirayə gəlirliliyi.

## Struktur

```
estate_price_ml/
  config.py      parametrlər, valyuta məzənnələri, xüsusiyyət siyahıları, seqmentlər
  data.py        SQLite bazadan və ya CSV faylından elanların oxunması
  features.py    təmizləmə, valyuta çevrilməsi, xüsusiyyətlərin qurulması
  segments.py    seqmentlərə bölmə (satış, kirayə, sahibkar, vasitəçi)
  model.py       PriceModel: emal, HistGradientBoosting, metrikalar
  analysis.py    bazar və bölgə analizi, kirayə gəlirliliyi
  train.py       bütün seqmentlərin öyrədilməsi və hesabat
  cli.py         əmr sətri: train, analyze, predict
tests/           sintetik data üzərində vahid testlər
```

## Quraşdırma

```
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Testlər üçün:

```
pip install pytest
pytest
```

## İstifadə

Model öyrətmək:

```
estate-price-ml train --db /yol/db.sqlite3 --out models
```

Bazar analizi:

```
estate-price-ml analyze --db /yol/db.sqlite3 --region district
```

Hazır modellə proqnoz:

```
estate-price-ml predict --model models/satis.joblib --csv yeni_elanlar.csv
```

Baza yolu `ESTATEPRICEML_DB` mühit dəyişəni ilə də verilə bilər.

## Xüsusiyyətlər

Numerik xüsusiyyətlər: sahə, otaq sayı, torpaq sahəsi, mərtəbə, mərtəbə nisbəti,
sahənin loqarifmi, koordinat mövcudluğu, baxış sayı, en və uzunluq.

Kateqorik xüsusiyyətlər: şəhər, rayon, qəsəbə, əmlak növü, mənbə.

Hədəf dəyişən qiymətdir. Bütün qiymətlər AZN valyutasına çevrilir və model
loqarifmik qiymət üzərində öyrədilir. Qiymət və sahə üzrə kənar dəyərlər seqment
daxilində budanır.

## Metrikalar

Hər model üçün MAE, RMSE, R2 və MAPE hesablanır. Nəticə `hesabat.json` faylına
yazılır.

## Qeyd

Model faylları, hesabatlar və baza faylları repozitoriyaya daxil edilmir.
Kitabxana yalnız kod və testlərdən ibarətdir.
