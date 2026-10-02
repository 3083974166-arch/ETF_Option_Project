import argparse
from pathlib import Path
from .data import probe, download_history, download_underlying, official_validate
from .pipeline import pipeline
from .backtest import Config


def main():
    parser = argparse.ArgumentParser(description="50ETF options: real data only; fixtures live in tests")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("probe", help="Check public endpoints; never fabricate market data")
    p.add_argument("--out", default="data/raw/probe")
    d = sub.add_parser("download", help="Download historical prices using verified contract mappings")
    d.add_argument("--contracts", required=True)
    d.add_argument("--out", default="data/raw/download")
    u = sub.add_parser("download-etf", help="Download raw ETF close and apply verified dividends")
    u.add_argument("--start", required=True)
    u.add_argument("--end", required=True)
    u.add_argument("--dividends", required=True)
    u.add_argument("--out", default="data/raw/download")
    v = sub.add_parser("validate-official", help="Validate against normalized exchange export")
    v.add_argument("--contracts", required=True)
    v.add_argument("--official", required=True)
    v.add_argument("--out", default="results/official_validation.csv")
    p = sub.add_parser("run", help="Run full real-data pipeline")
    p.add_argument("--input", default="data/raw/ready")
    p.add_argument("--out", default="results/real_run")
    p.add_argument("--split", required=True)
    p.add_argument("--proxy-costs", action="store_true")
    p.add_argument("--assume-etf-borrow", action="store_true")
    p.add_argument("--no-charts", action="store_true")
    args = parser.parse_args()
    if args.command == "probe":
        result = probe(args.out)
        for row in result["checks"]:
            print(row["name"], row["status"], row.get("rows", ""))
    elif args.command == "download":
        download_history(args.contracts, args.out)
    elif args.command == "download-etf":
        download_underlying(args.start, args.end, args.dividends, args.out)
    elif args.command == "validate-official":
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        print(official_validate(args.contracts, args.official, args.out).to_string(index=False))
    elif args.command == "run":
        config = Config(price_mode="proxy" if args.proxy_costs else "quotes", allow_short_etf=args.assume_etf_borrow)
        print(pipeline(args.input, args.out, args.split, config, charts=not args.no_charts))


if __name__ == "__main__":
    main()

