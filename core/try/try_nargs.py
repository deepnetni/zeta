import argparse


def parse():
    parser = argparse.ArgumentParser(
        description="compute the metrics score with input file or directory."
        "\n\nExample: python compute_metrics.py --src xx --out yy --sisnr ",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--metrics", help="compute multi-metrics", nargs="+", action="store_true")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse()
    print(args)
