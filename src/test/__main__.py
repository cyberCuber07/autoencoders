from src.test.utils import main
import argparse


if __name__ == "__main__":
    args = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    args.add_argument("--path", type=str, default=None)
    args.add_argument("--images", type=str, default="images")
    args.add_argument("--weights", type=str, default="weights/65.h5")
    args = args.parse_args()

    main(args.weights, args.images)
