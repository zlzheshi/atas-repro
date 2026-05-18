import open_clip


def main() -> None:
    checkpoint = "/mnt/t1b6/xuzhejia/checkpoints/open_clip/open_clip_pytorch_model.bin"
    model, _, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained=checkpoint)
    print("loaded", type(model).__name__)


if __name__ == "__main__":
    main()

