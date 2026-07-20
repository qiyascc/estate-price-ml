import argparse
import json

from estate_price_ml import analysis, data, features, train
from estate_price_ml.estimate import estimate
from estate_price_ml.model import PriceModel


def _show(obj):
    print(json.dumps(obj, ensure_ascii=False, indent=2))


def cmd_train(args):
    report = train.train_all(db_path=args.db, csv_path=args.csv, out_dir=args.out)
    _show(report["umumi"])


def cmd_analyze(args):
    frame = train.prepare(db_path=args.db, csv_path=args.csv)
    _show(
        {
            "umumi": analysis.market_summary(frame),
            "sahibkar_vasitechi": analysis.owner_vs_agent(frame),
            "bolge_uzre_qiymet": analysis.region_summary(frame, by=args.region).to_dict(
                orient="records"
            ),
            "kiraye_gelirliligi": analysis.rental_yield_by_region(frame).to_dict(orient="records"),
        }
    )


def cmd_predict(args):
    model = PriceModel.load(args.model)
    frame = data.load_with_metro(csv_path=args.csv)
    frame = features.clean(frame)
    frame = features.add_features(frame)
    predictions = model.predict(frame)
    for value in predictions:
        print(round(float(value), 2))


def cmd_estimate(args):
    record = {
        "deal_type": args.deal,
        "city": args.city,
        "district": args.district,
        "neighborhood": args.neighborhood,
        "metro": args.metro,
        "property_type": args.type,
        "rooms": args.rooms,
        "area": args.area,
        "floor_current": args.floor,
        "floor_total": args.floors,
        "is_daily_rent": args.daily,
    }
    value = estimate(args.model, record)[0]
    print(round(value))


def build_parser():
    parser = argparse.ArgumentParser(prog="estate-price-ml")
    sub = parser.add_subparsers(dest="command", required=True)

    trainer = sub.add_parser("train")
    trainer.add_argument("--db", default=None)
    trainer.add_argument("--csv", default=None)
    trainer.add_argument("--out", default="models")
    trainer.set_defaults(func=cmd_train)

    analyzer = sub.add_parser("analyze")
    analyzer.add_argument("--db", default=None)
    analyzer.add_argument("--csv", default=None)
    analyzer.add_argument("--region", default="district")
    analyzer.set_defaults(func=cmd_analyze)

    predictor = sub.add_parser("predict")
    predictor.add_argument("--model", required=True)
    predictor.add_argument("--csv", required=True)
    predictor.set_defaults(func=cmd_predict)

    single = sub.add_parser("estimate")
    single.add_argument("--model", required=True)
    single.add_argument("--deal", default="sale")
    single.add_argument("--city", default="")
    single.add_argument("--district", default="")
    single.add_argument("--neighborhood", default="")
    single.add_argument("--metro", default="")
    single.add_argument("--type", default="")
    single.add_argument("--rooms", type=int, default=0)
    single.add_argument("--area", type=float, default=0.0)
    single.add_argument("--floor", type=int, default=0)
    single.add_argument("--floors", type=int, default=0)
    single.add_argument("--daily", type=int, default=0)
    single.set_defaults(func=cmd_estimate)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
