import os
import config

OWLV2_ID = "google/owlv2-base-patch16-ensemble"
DINO_ID = "IDEA-Research/grounding-dino-tiny"


def first_run_setup():
    from transformers import (
        Owlv2Processor,
        Owlv2ForObjectDetection,
        AutoProcessor,
        AutoModelForZeroShotObjectDetection,
    )

    os.makedirs(config.MODEL_DIR, exist_ok=True)
    owl_path = os.path.join(config.MODEL_DIR, "owlv2")
    dino_path = os.path.join(config.MODEL_DIR, "grounding_dino")

    if not os.path.isdir(owl_path):
        print("Downloading OWLv2 model (one-time)...")
        proc = Owlv2Processor.from_pretrained(OWLV2_ID)
        model = Owlv2ForObjectDetection.from_pretrained(OWLV2_ID)
        proc.save_pretrained(owl_path)
        model.save_pretrained(owl_path)

    if not os.path.isdir(dino_path):
        print("Downloading Grounding DINO model (one-time)...")
        proc = AutoProcessor.from_pretrained(DINO_ID)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(DINO_ID)
        proc.save_pretrained(dino_path)
        model.save_pretrained(dino_path)

    if not os.path.isfile(config.QUERY_FILE):
        open(config.QUERY_FILE, "w").close()

    return owl_path, dino_path
