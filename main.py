"""Face analysis demo: age, gender and emotion from an image, a video or the webcam.

    python main.py --image path/to/photo.jpg [--output result.jpg]
    python main.py --video path/to/clip.mp4 [--output result.mp4]
    python main.py --webcam [--camera 0]
"""
import argparse
import os
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2  # noqa: E402

from utils.predictor import FaceAnalyzer, draw_results  # noqa: E402


def print_results(results):
    if not results:
        print("No faces detected.")
    for i, r in enumerate(results, 1):
        print(f"Face {i} at {r['box']}: age {r['age']:.1f}, {r['gender']} ({r['gender_confidence']:.2f}), "
              f"{r['emotion']} ({r['emotion_confidence']:.2f})")


def run_image(analyzer, path, output, show):
    image = cv2.imread(path)
    if image is None:
        raise SystemExit(f"Cannot read image: {path}")
    results = analyzer.analyze(image)
    print_results(results)
    vis = draw_results(image, results)
    if output:
        cv2.imwrite(output, vis)
        print(f"Saved {output}")
    if show:
        cv2.imshow("Face analysis", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def run_stream(analyzer, source, output, show):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video source: {source}")
    writer = None
    prev = time.time()
    print("Press 'q' to quit, 's' to save a screenshot.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        vis = draw_results(frame, analyzer.analyze(frame))
        now = time.time()
        cv2.putText(vis, f"{1.0 / max(now - prev, 1e-6):.1f} FPS", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0), 2)
        prev = now
        if output:
            if writer is None:
                fps = cap.get(cv2.CAP_PROP_FPS) or 25
                writer = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*"mp4v"), fps, vis.shape[1::-1])
            writer.write(vis)
        if show:
            cv2.imshow("Face analysis", vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                name = f"screenshot_{int(now)}.jpg"
                cv2.imwrite(name, vis)
                print(f"Saved {name}")
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", help="path to an image")
    src.add_argument("--video", help="path to a video file")
    src.add_argument("--webcam", action="store_true", help="use a webcam")
    parser.add_argument("--camera", type=int, default=0, help="webcam index")
    parser.add_argument("--output", help="where to save the annotated image/video")
    parser.add_argument("--no-show", action="store_true", help="don't open a window (e.g. on a server)")
    parser.add_argument("--no-tta", action="store_true", help="disable flip test-time augmentation (faster)")
    args = parser.parse_args()

    analyzer = FaceAnalyzer(tta=not args.no_tta)
    show = not args.no_show
    if args.image:
        run_image(analyzer, args.image, args.output, show)
    else:
        run_stream(analyzer, args.camera if args.webcam else args.video, args.output, show)


if __name__ == "__main__":
    main()
