import warnings

import argparse
from config import get_cfg_defaults  # pylint: disable=import-error
from yacs.config import CfgNode as CN
import logging

def get_evaluation_mode(eval_mode):
    """Gets evaluation mode from the config file. This is used to determine whether to use how to process the model output to get the final coords.

    Args:
        eval_mode (str): string for evaulation mode
        og_im_size [int, int]: original image size
        inp_size [int, int]: image resized for input to network (can be same as og_im_size)

    Raises:
        ValueError: if eval_mode is not supported

    Returns:
        bool, bool: settings for the evaluation modes.
    """

    # Evaluate on input size to network, using coordinates resized to the input size
    if eval_mode == "use_input_size":
        use_full_res_coords = False
        resize_first = False
    # Scale model predicted sized heatmap up to full resolution and then obtain coordinates (recommended)
    elif eval_mode == "scale_heatmap_first":
        use_full_res_coords = True
        resize_first = True
    # Obtain coordinates from input sized heatmap and scale up the coordinates to full sized heatmap.
    elif eval_mode == "scale_pred_coords":
        use_full_res_coords = True
        resize_first = False
    else:
        raise ValueError(
            "value for cg.INFERENCE.EVALUATION_MODE not recognised. Choose from: scale_heatmap_first, scale_pred_coords, use_input_size"
        )
    return use_full_res_coords, resize_first


def infer_additional_arguments(yaml_args):
    """Uses the config file to infer additional arguments that are not explicitly defined in the config file.

    Args:
        yaml_args (.yaml): _description_

    Raises:
        NotImplementedError: _description_

    Returns:
        _type_: _description_
    """
    yaml_args.INFERRED_ARGS = CN()
    yaml_args.INFERRED_ARGS.GEN_HM_IN_MAINTHREAD = False

    



def arg_parse():
    """Parses shell arguments, loads the default config file and merges it with user defined arguments. Calls the argument checker.
    Returns:
        config: config for the programme
    """
    parser = argparse.ArgumentParser(description="PyTorch Landmark Localization U-Net")
    parser.add_argument("--cfg", required=True, help="path to config file", type=str)
    parser.add_argument(
        "--gpus",
        default=1,
        help="gpu id(s) to use. None/int(0) for cpu. list[x,y] for xth, yth GPU."
        "str(x) for the first x GPUs. str(-1)/int(-1) for all available GPUs",
    )
    parser.add_argument("--fold", type=int)
    args = parser.parse_args()

    # ---- setup configs ----
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)

    if args.fold:
        cfg.TRAINER.FOLD = args.fold
    # cfg.freeze()
    cfg = infer_additional_arguments(cfg)

    logging.info("Config: \n ", cfg)

    return cfg
