import tensorflow as tf
import os
import logging
import numpy as np
from utils.data_loading import load_datalist, get_datatype_load
from utils.image_utils import load_and_resize_image

class LandmarkDataset(tf.keras.utils.Sequence):
    
    def __init__(
        self,
        landmarks,
        sigmas,
        LabelGenerator,
        hm_lambda_scale: float,
        annotation_path: str,
        split: str = "training",
        root_path: str = "./data",
        cv: int = -1,
        cache_data: bool = False,
        debug: bool = False,
        sample_patch_size=[512, 512],
        sample_patch_from_resolution=[512, 512],
        num_res_supervisions: int = 1,
        data_augmentation_strategy: str = None,
        data_augmentation_package: str = None,
        dataset_split_size: int = -1,
        additional_sample_attribute_keys=[],
    ):
        
        self.landmarks = landmarks
        self.sigmas = sigmas
        self.LabelGenerator = LabelGenerator
        self.hm_lambda_scale = hm_lambda_scale
        self.annotation_path = annotation_path
        self.split = split
        self.root_path = root_path
        self.cv = cv
        self.cache_data = cache_data
        self.debug = debug
        self.sample_patch_size = sample_patch_size
        self.sample_patch_from_resolution = sample_patch_from_resolution
        self.num_res_supervisions = num_res_supervisions
        self.data_augmentation_package = data_augmentation_package
        self.data_augmentation_strategy = data_augmentation_strategy
        self.dataset_split_size = dataset_split_size
        self.additional_sample_attribute_keys = additional_sample_attribute_keys

        self.additional_sample_attributes = {
            k: [] for k in self.additional_sample_attribute_keys
        }
        
        # Lists to save the image paths (or images if caching), target coordinates (scaled to input size), and full resolution coords.
        self.images = []
        self.target_coordinates = []
        self.full_res_coordinates = [] # full_res will be same as target if input and original image same size
        self.image_paths = []
        self.uids = []
        self.annotation_available = []
        self.original_image_sizes = []
        self.image_resizing_factors = []

        #Load in dataset
        # We are using cross-validation, following our convention of naming each json train with the append "foldX" where (X= self.cv)
        if cv >= 0:
            label_std = os.path.join("fold" + str(self.cv) + ".json")
            logging.info("Loading %s data for CV %s "% (self.split, os.path.join(self.annotation_path, label_std)))
            datalist = load_datalist(
                os.path.join(self.annotation_path, label_std),
                data_list_key=self.split,
                base_dir=self.root_path,
            )
            
            #Get only a certain number of datapoints if dataset_split_size is defined (useful for debugging).
            if self.dataset_split_size != -1:
                datalist = datalist[: self.dataset_split_size]
        # Not using CV, load the specified json file
        else:
            logging.info("Loading %s data (no CV) for %s " % (self.split, self.annotation_path))
            datalist = load_datalist(
                self.annotation_path, data_list_key=self.split, base_dir=self.root_path
            )

        #Get dataloading function dependent on file type
        self.datatype_load = get_datatype_load(datalist[0]["image"])
        self.load_function = lambda img: img


        #Go through dataset and cache or format paths
        for idx, data in enumerate(datalist):
            ### Add coordinate labels as sample attribute, if annotations available
            # case when data has no annotation, i.e. inference only, just set target coords to 0,0 and annotation_available to False
            if (not isinstance(data["coordinates"], list)) or (
                "has_annotation" in data.keys() and data["has_annotation"] == False
            ):
                interested_landmarks = np.array([[0, 0]] * len(self.landmarks))
                self.annotation_available.append(False)
                self.full_res_coordinates.append(interested_landmarks)

                if self.split == "training" or self.split == "validation":
                    raise ValueError(
                        "Training/Validation data must have annotations. Check your data. Sample that failed: ",
                        data,
                    )
            else:
                interested_landmarks = np.array(data["coordinates"])[self.landmarks, :2]
                self.full_res_coordinates.append(np.array(data["coordinates"])[self.landmarks, :2])
                self.annotation_available.append(True)

 
                # Not caching, so add image path.
                self.images.append(data["image"])  # just appends the path, the load_function will load it later.

            self.target_coordinates.append(interested_landmarks)
            self.image_paths.append(data["image"])
            self.uids.append(data["id"])

            # Extended dataset class can add more attributes to each sample here
            self.add_additional_sample_attributes(data)
        
        #Test that all uids are unique
        non_unique = [
            [x, self.image_paths[x_idx]]
            for x_idx, x in enumerate(self.uids)
            if self.uids.count(x) > 1
        ]
        assert len(non_unique) == 0, (
            "Not all uids are unique! Check your data. %s non-unqiue uids from %s samples , they are: %s \n "
            % (len(non_unique), len(self.uids), non_unique)
        )

    def __getitem__(self, index):
        """Main function of the dataloader. Gets a data sample.



        Args:
            index (_type_): _description_

        Returns:
            It must return a dictionary with the keys:
            sample = {
                "image" (torch.tensor, shape (1, H, W)): tensor of input image to network.
                "label" (Dict of torch.tensors):  if self.generate_hms_here bool -> (Dictionary of labels e.g. tensor heatmaps, see LabelGenerator for details); else -> [].
                "target_coords (np.array, shape (num_landmarks, 2))": list of target coords of for landmarks, same scale as input to model.
                "landmarks_in_indicator" (list of 1s and/or 0s, shape (1, num_landmarks)): list of bools, whether the landmark is in the image or not.
                "full_res_coords" (np.array, shape (num_landmarks, 2)): original list of coordinates of shape ,same scale as original image (so may be same as target_coords)
                "image_path" (str): path to image, from the JSON file.
                "uid" (str): sample's unique id, from the JSON file.
                "annotation_available" (bool): Whether the JSON file provided annotations for this sample (necessary for training and validation).
                "resizing_factor" (np.array, shape (1,2)): The x and y scale factors that the image was resized by to fit the network input size.
                "original_image_size" (np.array, shape (2,1)): The resolution of the original image before it was resized.
                ANY EXTRA ATTRIBUTES ADDED BY add_additional_sample_attributes()
             }


        """


        hm_sigmas = self.sigmas
        # print("load time: " , (time()-sorgin))
        coords = self.target_coordinates[index]
        full_res_coods = self.full_res_coordinates[index]
        im_path = self.image_paths[index]
        run_time_debug = False
        this_uid = self.uids[index]
        is_annotation_available = self.annotation_available[index]
        x_y_corner = [0, 0]
        image = self.load_function(self.images[index])

        # If we cached the data, we don't need to get original image size. If not, we need to load it here.
        if self.cache_data:
            resized_factor = self.image_resizing_factors[index]
            original_size = self.original_image_sizes[index]

        else:
            resized_factor, original_size, image, coords = self.load_and_resize_image(
                image, coords
            )

        untransformed_coords = coords

        untransformed_im = image

       label = self.LabelGenerator.generate_labels(
                input_coords,
                x_y_corner,
                landmarks_in_indicator,
                self.heatmap_label_size,
                hm_sigmas,
                self.num_res_supervisions,
                self.hm_lambda_scale,
            )

        input_coords = coords
        input_image = torch.from_numpy(image).float()
        landmarks_in_indicator = [1 for xy in input_coords]

      

        sample = {
            "image": input_image,
            "label": label,
            "target_coords": input_coords,
            "landmarks_in_indicator": landmarks_in_indicator,
            "full_res_coords": full_res_coods,
            "image_path": im_path,
            "uid": this_uid,
            "annotation_available": is_annotation_available,
            "resizing_factor": resized_factor,
            "original_image_size": original_size,
        }

        # add additional sample attributes from child class.
        for key_ in list(self.additional_sample_attributes.keys()):
            sample[key_] = self.additional_sample_attributes[key_][index]

        if self.debug or run_time_debug:
            print("sample: ", sample)
            self.LabelGenerator.debug_sample(
                sample, untransformed_im, untransformed_coords
            )
        return sample


    def sample_patch(self, image, landmarks, lm_safe_region=0, safe_padding=128):
        """Samples a patch from the image. It ensures a landmark is in a patch with a self.sample_patch_bias% chance.
            The patch image is larger than the patch-size by safe_padding on every side for safer data augmentation.
            Therefore, the image is first padded with zeros on each side to stop out of bounds when sampling from the edges.

        Args:
            image (_type_): image to sample
            landmarks (_type_): list of landmarks
            lm_safe_region (int, optional): # pixels away from the edge the landmark must be to count as "in" the patch . Defaults to 0.
            safe_padding (int, optional): How much bigger on each edge the patch should be for safer data augmentation . Defaults to 128.

        Returns:
            _type_: cropped padded sample
            landmarks normalised to within the patch
            binary indicator of which landmarks are in the patch.

        """

        z_rand = np.random.uniform(0, 1)
        landmarks_in_indicator = []
        if z_rand >= (1 - self.sample_patch_bias):

            # Keep sampling until landmark is in patch
            while 1 not in landmarks_in_indicator:
                landmarks_in_indicator = []

                #
                y_rand = np.random.randint(
                    0, self.load_im_size[1] - self.sample_patch_size[1]
                )
                x_rand = np.random.randint(
                    0, self.load_im_size[0] - self.sample_patch_size[0]
                )

                for lm in landmarks:
                    landmark_in = 0

                    # Safe region means landmark is not right on the edge
                    if (
                        y_rand + lm_safe_region
                        <= lm[1]
                        <= (y_rand + self.sample_patch_size[1]) - lm_safe_region
                    ):
                        if (
                            x_rand + lm_safe_region
                            <= lm[0]
                            <= (x_rand + self.sample_patch_size[0]) - lm_safe_region
                        ):
                            landmark_in = 1

                    landmarks_in_indicator.append(landmark_in)

                # Tested with the extremes, its all ok.
                # y_rand = self.load_im_size[1]-self.sample_patch_size[1]
                # x_rand = self.load_im_size[0]-self.sample_patch_size[0]
                # y_rand = 0
                # x_rand = 0
                # y_rand = safe_padding
                # x_rand = self.load_im_size[0]-self.sample_patch_size[0]

        else:
            y_rand = np.random.randint(
                0, self.load_im_size[1] - self.sample_patch_size[1]
            )
            x_rand = np.random.randint(
                0, self.load_im_size[0] - self.sample_patch_size[0]
            )

            for lm in landmarks:
                landmark_in = 0
                if (
                    y_rand + lm_safe_region
                    <= lm[1]
                    <= y_rand + self.sample_patch_size[1] - lm_safe_region
                ):
                    if (
                        x_rand + lm_safe_region
                        <= lm[0]
                        <= (x_rand + self.sample_patch_size[0]) - lm_safe_region
                    ):
                        landmark_in = 1
                landmarks_in_indicator.append(landmark_in)

        # Add the safe padding size
        y_rand_safe = y_rand + safe_padding
        x_rand_safe = x_rand + safe_padding

        # First pad image
        padded_image = np.expand_dims(
            np.pad(image[0], (safe_padding, safe_padding)), axis=0
        )
        padded_patch_size = [x + (2 * safe_padding) for x in self.sample_patch_size]

        # We pad before and after the slice.
        y_rand_pad = y_rand_safe - safe_padding
        x_rand_pad = x_rand_safe - safe_padding
        cropped_padded_sample = padded_image[
            :,
            y_rand_pad : y_rand_pad + padded_patch_size[1],
            x_rand_pad : x_rand_pad + padded_patch_size[0],
        ]

        # Calculate the new origin: 2*safe_padding bc we padded image & then added pad to the patch.
        normalized_landmarks = [
            [
                (lm[0] + 2 * safe_padding) - (x_rand_safe),
                (lm[1] + 2 * safe_padding) - (y_rand_safe),
            ]
            for lm in landmarks
        ]

        if self.debug:
            padded_lm = [
                [lm[0] + safe_padding, lm[1] + safe_padding] for lm in landmarks
            ]

            print(
                "\n \n \n the min xy is [%s,%s]. padded is [%s, %s] normal landmark is %s, padded lm is %s \
             and the normalized landmark is %s : "
                % (
                    y_rand_safe,
                    x_rand_safe,
                    x_rand_pad,
                    y_rand_pad,
                    landmarks,
                    padded_lm,
                    normalized_landmarks,
                )
            )

            visualize_patch(
                image[0],
                landmarks[0],
                padded_image[0],
                padded_lm[0],
                cropped_padded_sample[0],
                normalized_landmarks[0],
                [x_rand_pad, y_rand_pad],
            )
        return (
            cropped_padded_sample,
            normalized_landmarks,
            landmarks_in_indicator,
            [x_rand, y_rand],
        )

    def on_epoch_end(self):
        pass
    

    
    def __len__(self):
        return len(self.images)


    def add_additional_sample_attributes(self, data):
        """
        Add more attributes to each sample.

        """
        for k_ in self.additional_sample_attribute_keys:
            keyed_data = data[k_]
            self.additional_sample_attributes[k_].append(keyed_data)