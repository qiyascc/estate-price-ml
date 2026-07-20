import json
import os
import time

from estate_price_ml import analysis, config, data, features, segments
from estate_price_ml.model import PriceModel


def prepare(db_path=None, csv_path=None):
    frame = data.load_with_metro(db_path=db_path, csv_path=csv_path)
    frame = features.clean(frame)
    frame = features.add_features(frame)
    frame = features.winsorize_target(frame, by="deal_type")
    return frame


def train_all(db_path=None, csv_path=None, out_dir="models", verbose=True):
    os.makedirs(out_dir, exist_ok=True)
    started = time.time()
    frame = prepare(db_path=db_path, csv_path=csv_path)
    if verbose:
        print("hazir data:", len(frame), "elan")
    report = {
        "elan_sayi": int(len(frame)),
        "umumi": analysis.market_summary(frame),
        "emlak_novleri": analysis.property_type_breakdown(frame).to_dict(orient="records"),
        "sahibkar_vasitechi": analysis.owner_vs_agent(frame),
        "kiraye_gelirliligi": analysis.rental_yield_by_region(frame).to_dict(orient="records"),
        "segmentler": {},
    }
    for name in config.SEGMENTS:
        part = segments.split_segment(frame, name)
        if len(part) < config.MIN_SEGMENT_ROWS:
            report["segmentler"][name] = {"elan_sayi": int(len(part)), "status": "az_data"}
            if verbose:
                print("seqment", name, "az data:", len(part))
            continue
        if verbose:
            print("seqment", name, "oyredilir:", len(part), "elan")
        model = PriceModel()
        holdout = model.evaluate(part)
        cv = model.cross_validate(part)
        model.fit(part)
        model.save(os.path.join(out_dir, "{}.joblib".format(name)))
        report["segmentler"][name] = {
            "elan_sayi": int(len(part)),
            "holdout_metrikalar": holdout,
            "cv_metrikalar": cv,
            "bazar": analysis.market_summary(part),
            "bolge_uzre_qiymet": analysis.region_summary(part).to_dict(orient="records"),
            "metro_uzre_qiymet": analysis.region_summary(part, by="metro").to_dict(orient="records"),
        }
        if verbose:
            print("  holdout mae:", round(holdout["mae"], 1), "r2:", round(holdout["r2"], 3),
                  "mape:", round(holdout["mape"], 1))
    report["saniye"] = round(time.time() - started, 1)
    with open(os.path.join(out_dir, "hesabat.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    if verbose:
        print("bitdi:", report["saniye"], "saniye")
    return report
