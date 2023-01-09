import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import matplotlib.patches as patchesplt


class LabelGenerator(ABC):
    """Super class that defines some methods for generating landmark labels."""

    def __init__(self, full_res_size, network_input_size):
        self.full_res_size = full_res_size
        self.network_input_size = network_input_size

    @abstractmethod
    def generate_labels(
        self,
        landmarks,
        x_y_corner_patch,
        landmarks_in_indicator,
        input_size,
        hm_sigmas,
        num_res_supervisions,
        hm_lambda_scale,
    ):
        """generates heatmaps for given landmarks of size input_size, using sigma hm_sigmas.
            Generates int(num_res_supervisions) heatmaps, each half the size as previous.
            The hms are scaled by float hm_lambda_scale

        Args:
            landmarks [[int, int]]: list of landmarks to gen heatmaps for
            input_size [int, int]: size of first heatmap
            hm_sigmas [float]: gaussian sigmas of heatmaps, 1 for each landmark
            num_res_supervisions int: number of heatmaps to generate, each half resolution of previous.
            hm_lambda_scale float: value to scale heatmaps by.


            landmarks ([[int,int]]): A 2D list of ints where each entry is the [x,y] coordinate of a landmark.
            x_y_corner_patch ([int, int]): The coordinates of the top left of the image sample you are creating a heatmap for.
            landmarks_in_indicator ([int]): A list of 1s and 0s where 1 indicates the landmark was in the model input image and 0 if not.
            image_size ([int, int]): Size of the heatmap to produce.
            sigmas ([float]): List of sigmas for the size of the heatmap. Each sigma is for the heatmap for a level of deep supervision.
                            The first sigma defines the sigma for the full-size resolution heatmap, the next for the half-resolution heatmap,
                            the next for the 1/8 resolution heatmap etc.
            num_res_levels (int): Number of deep supervision levels (so should be the length of the list of sigmas. Kind-of redundant).
            lambda_scale (float): Scaler to multiply the heatmap magnitudes by.
            dtype: datatype of the output label
            to_tensor (bool): Whether to output the label as a tensor object or numpy object.

        """


class UNetLabelGenerator(LabelGenerator):
    """Generates target heatmaps for the U-Net network training scheme"""

    def __init__(self):
        super(LabelGenerator, self).__init__()

    def generate_labels(
        self,
        landmarks,
        x_y_corner_patch,
        landmarks_in_indicator,
        input_size,
        hm_sigmas,
        num_res_supervisions,
        hm_lambda_scale=100,
        dtype=np.float32,
        to_tensor=True,
    ):
        """Generates Gaussian heatmaps for given landmarks of size input_size, using sigma hm_sigmas."""

        return_dict = {"heatmaps": []}

        heatmap_list = []
        resizing_factors = [[2**x, 2**x] for x in range(num_res_supervisions)]

        # Generates a heatmap for multiple resolutions based on # down steps in encoder (1x, 0.5x, 0.25x etc)
        for size_f in resizing_factors:
            intermediate_heatmaps = []
            # Generate a heatmap for each landmark
            for idx, lm in enumerate(landmarks):

                lm = np.round(lm / size_f)
                downsample_size = [input_size[0] / size_f[0], input_size[1] / size_f[1]]
                down_sigma = hm_sigmas[idx] / size_f[0]

                # If the landmark is present in image, generate a heatmap, otherwise generate a blank heatmap.
                if landmarks_in_indicator[idx] == 1:
                    intermediate_heatmaps.append(
                        gaussian_gen(lm, downsample_size, 1, down_sigma, dtype, hm_lambda_scale)
                    )
                else:
                    intermediate_heatmaps.append(np.zeros((int(downsample_size[0]), int(downsample_size[1]))))
            heatmap_list.append(np.array(intermediate_heatmaps))

        hm_list = heatmap_list[::-1]

        if to_tensor:
            all_seg_labels = []
            for maps in hm_list:
                all_seg_labels.append(tf.Variable.from_numpy(maps).float())

            hm_list = all_seg_labels

        return_dict["heatmaps"] = hm_list
        return return_dict

    def debug_prediction(
        self,
        input_dict,
        prediction_output,
        predicted_coords,
        input_size_pred_coords,
        logged_vars,
        extra_info,
    ):
        """Visually debug a prediction and compare to the target. Provide logging and visualisation of the sample."""
        heatmap_label = input_dict["label"]["heatmaps"][
            -1
        ]  # -1 to get the last layer only (ignore deep supervision labels)

        transformed_targ_coords = np.array(input_dict["target_coords"])
        full_res_coords = np.array(input_dict["full_res_coords"])
        transformed_input_image = input_dict["image"]

        predicted_heatmap = [x.cpu().detach().numpy() for x in prediction_output][
            -1
        ]  # -1 to get the last layer only (ignore deep supervision predictions)

        predicted_coords = [x.cpu().detach().numpy() for x in predicted_coords]
        input_size_pred_coords = extra_info["coords_og_size"]

        for sample_idx, ind_sample in enumerate(logged_vars):
            print("\n uid: %s. Mean Error: %s " % (ind_sample["uid"], ind_sample["Error All Mean"]))
            colours = np.arange(len(predicted_coords[sample_idx]))

            # Only show debug if any landmark error is >10 pixels!
            if (
                len(
                    [
                        x
                        for x in range(len(predicted_coords[sample_idx]))
                        if (ind_sample["L" + str(x)] != None and ind_sample["L" + str(x)] > 10)
                    ]
                )
                > 0
            ):
                fig, ax = plt.subplots(1, ncols=1, squeeze=False)

                for coord_idx, pred_coord in enumerate(predicted_coords[sample_idx]):
                    print(
                        "L%s: Full Res Prediction: %s, Full Res Target: %s, Error: %s. Input Res targ %s, input res pred %s."
                        % (
                            coord_idx,
                            pred_coord,
                            full_res_coords[sample_idx][coord_idx],
                            ind_sample["L" + str(coord_idx)],
                            transformed_targ_coords[sample_idx][coord_idx],
                            input_size_pred_coords[sample_idx][coord_idx],
                        )
                    )

                    # difference between these is removing the padding (so -128, or whatever the patch padding was)
                    print(
                        "predicted (red) vs  target coords (green): ",
                        input_size_pred_coords[sample_idx][coord_idx],
                        transformed_targ_coords[sample_idx][coord_idx],
                    )

                    # 1) show input image with target lm and predicted lm
                    # 2) show predicted heatmap
                    # 3 show target heatmap

                    # 1)
                    ax[0, 0].imshow(transformed_input_image[sample_idx][0])
                    rect1 = patchesplt.Rectangle(
                        (
                            transformed_targ_coords[sample_idx][coord_idx][0],
                            transformed_targ_coords[sample_idx][coord_idx][1],
                        ),
                        6,
                        6,
                        linewidth=2,
                        edgecolor="g",
                        facecolor="none",
                    )
                    ax[0, 0].add_patch(rect1)
                    rect2 = patchesplt.Rectangle(
                        (
                            input_size_pred_coords[sample_idx][coord_idx][0].detach().numpy(),
                            input_size_pred_coords[sample_idx][coord_idx][1].detach().cpu().numpy(),
                        ),
                        6,
                        6,
                        linewidth=2,
                        edgecolor="pink",
                        facecolor="none",
                    )
                    ax[0, 0].add_patch(rect2)

                    ax[0, 0].text(
                        transformed_targ_coords[sample_idx][coord_idx][0],
                        transformed_targ_coords[sample_idx][coord_idx][1] + 10,  # Position
                        "L" + str(coord_idx),  # Text
                        verticalalignment="bottom",  # Centered bottom with line
                        horizontalalignment="center",  # Centered with horizontal line
                        fontsize=12,  # Font size
                        color="g",  # Color
                    )
                    if ind_sample["L" + str(coord_idx)] > 10:
                        pred_text = "r"
                    else:
                        pred_text = "pink"
                    ax[0, 0].text(
                        input_size_pred_coords[sample_idx][coord_idx][0].detach().cpu().numpy(),
                        input_size_pred_coords[sample_idx][coord_idx][1].detach().cpu().numpy() + 10,  # Position
                        "L" + str(coord_idx) + " E=" + str(np.round(ind_sample["L" + str(coord_idx)], 2)),  # Text
                        verticalalignment="bottom",  # Centered bottom with line
                        horizontalalignment="center",  # Centered with horizontal line
                        fontsize=12,  # Font size
                        color=pred_text,  # Color
                    )
                    ax[0, 0].set_title(
                        "uid: %s. Mean Error: %s +/- %s"
                        % (
                            ind_sample["uid"],
                            np.round(ind_sample["Error All Mean"], 2),
                            np.round(ind_sample["Error All Std"]),
                        )
                    )

                plt.show()
                plt.close()


# generate Guassian with center on landmark. sx and sy are the std.
def gaussian_gen(landmark, resolution, step_size, std, dtype=np.float32, lambda_scale=100):

    sx = std
    sy = std

    x = resolution[0] / step_size
    y = resolution[1] / step_size

    mx = landmark[0] / step_size
    my = landmark[1] / step_size

    x = np.arange(x)
    y = np.arange(y)

    x, y = np.meshgrid(x, y)  # get 2D variables instead of 1D

    # define guassian
    g = (
        (1)
        / (2.0 * np.pi * sx * sy)
        * np.exp(-((x - mx) ** 2.0 / (2.0 * sx**2.0) + (y - my) ** 2.0 / (2.0 * sy**2.0)))
    )

    # normalise between 0 and 1
    g *= 1.0 / g.max() * lambda_scale

    g[g <= 0] = -1

    return g
