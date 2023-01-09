import os

from utils.logging import setup_logger
from utils.setup import arg_parse
from utils.label_generator import UNetLabelGenerator

def main():
    cfg = arg_parse()
    
    #Setup output logger
    os.makedirs(cfg.OUTPUT.OUTPUT_DIR, exist_ok=True)
    setup_logger(os.path.join(cfg.OUTPUT.OUTPUT_DIR, cfg.OUTPUT.OUTPUT_LOG_NAME))

    #Setup label generator for target heatmaps
    heatmap_generator = UNetLabelGenerator(cfg.SAMPLER.RESOLUTION_TO_SAMPLE_FROM, cfg.PATCH_SIZE)

    dataset

if __name__ == "__main__":
    main()
