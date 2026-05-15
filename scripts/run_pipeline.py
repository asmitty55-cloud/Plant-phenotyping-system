from pt.core.capture import capture_images
from pt.core.processing import preprocess
from pt.core.analysis import analyze

def main():
    print("Starting phenotyping pipeline...")

    frames = capture_images()
    processed = preprocess(frames)
    results = analyze(processed)

    print("Done:", results)

if __name__ == "__main__":
    main()