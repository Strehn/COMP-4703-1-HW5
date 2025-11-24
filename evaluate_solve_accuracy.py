"""
Evaluate CAPTCHA Solving Accuracy

Uses solve_captchas_with_model.py logic to compute accuracy over a random subset
of CAPTCHA images. Intended as a quick validation tool for the homework.
"""
import os
import argparse
import numpy as np
from imutils import paths


def main():
    parser = argparse.ArgumentParser(description="Evaluate CAPTCHA solving accuracy")
    parser.add_argument("--image-folder", required=True, help="Folder containing CAPTCHA images")
    parser.add_argument("--samples", type=int, default=50, help="Number of random images to evaluate")
    args = parser.parse_args()

    captcha_image_files = list(paths.list_images(args.image_folder))
    if len(captcha_image_files) == 0:
        raise SystemExit(f"No images found in {args.image_folder}")

    if len(captcha_image_files) < args.samples:
        print(f"[WARN] Requested {args.samples} samples but only {len(captcha_image_files)} available. Using all.")
        sample_size = len(captcha_image_files)
    else:
        sample_size = args.samples

    captcha_image_files = np.random.choice(captcha_image_files, size=(sample_size,), replace=False)

    # Call the solver as a module import to reuse its logic
    # We avoid re-implementing segmentation and prediction here for simplicity.
    import solve_captchas_with_model as solver

    num_total = 0
    num_correct = 0
    num_skipped = 0

    for image_file in captcha_image_files:
        try:
            pred, skipped = solver.solve_single_image(image_file)  # type: ignore[attr-defined]
        except AttributeError:
            raise SystemExit(
                "solve_captchas_with_model.py must expose a function solve_single_image(path) -> (prediction:str|None, skipped:bool).\n"
                "Please add this helper to support programmatic evaluation."
            )

        if skipped:
            num_skipped += 1
            continue

        num_total += 1
        gt = os.path.splitext(os.path.basename(image_file))[0]
        if pred == gt:
            num_correct += 1

    if num_total > 0:
        acc = num_correct / num_total
        print(f"[REPORT] Evaluated: {num_total}  Correct: {num_correct}  Accuracy: {acc:.3f}  Skipped: {num_skipped}")
    else:
        print("[REPORT] No images evaluated.")


if __name__ == "__main__":
    main()
