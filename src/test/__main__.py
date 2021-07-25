from src.test.utils import load_imgs, load_model
from src.params import PREDICT_DIR
from src.utils import get_name
import argparse
import cv2


def main(model_weights, imgs_path):
    model = load_model(model_weights)
    imgs = load_imgs(imgs_path)

    predictions = model.predict(imgs)
    for prediciton in predictions:
        prediciton = prediciton * 255.
        cv2.imwrite(get_name(PREDICT_DIR, ".jpg"), prediciton)


if __name__ == "__main__":
    args = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    args.add_argument("--path", type=str, default=None)
    args.add_argument("--images", type=str, default="images")
    args.add_argument("--weights", type=str, default="weights/65.h5")
    args = args.parse_args()

    main(args.weights, args.images)
