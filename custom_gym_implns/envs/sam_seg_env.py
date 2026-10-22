import numpy as np
import glob
import re
import cv2
import gymnasium as gym
from gymnasium import spaces
from scipy.ndimage import gaussian_filter, map_coordinates

import sys
import os
sys.path.append(os.path.abspath(os.getcwd()))

from custom_gym_implns.envs.utils.repvit_sam_wrapper import RepVITSamWrapper
from datasets import get_dataset


class SamSegEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 1}

    def __init__(self, img_shape, embedding_shape, mask_shape, render_frame_shape,
                 max_steps, target_categories, dataset_config, sam_ckpt_fp, 
                 penalize_for_wrong_input=True, use_dice_score=False,
                 img_patch_size=None, num_patches=None, render_mode='rgb_array',
                 augmentation=None, tau_acc=0.5):
        
        assert len(img_shape) == 3, "Image shape should be (H, W, C)"
        assert len(embedding_shape) == 3, "Embedding shape should be (C, H, W)"
        assert len(mask_shape) == 2, "Mask shape should be (H, W)"
        assert len(render_frame_shape) == 2, "Render frame shape should be (H, W)"
        assert max_steps is None or max_steps > 0, "Max steps should be None or > 0"
        
        assert (img_patch_size is not None and num_patches is None) or \
            (img_patch_size is None and num_patches is not None), \
            "Either img_patch_size or num_patches should be provided, not both"

        assert render_mode in self.metadata["render_modes"], "Invalid render mode"


        self.img_shape = img_shape  # The size of the image (HxWxC)
        self.embedding_shape = embedding_shape  # The size of the SAM image encoder output
        self.mask_shape = mask_shape  # The size of the mask (HxW)
        self.render_frame_shape = render_frame_shape  # The size of the frame to render

        self.max_steps = max_steps  # The maximum number of steps the agent can take
        self.penalize_for_wrong_input = penalize_for_wrong_input  # Whether to penalize for wrong input
        self.use_dice_score = use_dice_score

        self.target_categories = target_categories  # The categories to segment
        self.dataset_config = dataset_config
        self.dataset = get_dataset(self.dataset_config)

        self.img_patch_size = img_patch_size
        
        # Data augmentation config (anti-overfitting)
        self.augmentation = augmentation if augmentation is not None else {'enabled': False}
        # Terminal acceptability threshold: r_T += (2*y_T - 1), y_T = 1[Dice(M_T,G) >= tau_acc]
        self.tau_acc = tau_acc

        self._image = None
        self._curr_target_cat = None
        self._sam_image_embeddings = None
        self._sam_pred_mask_prob = None
        self._sam_pred_mask = None
        self._categorical_instance_masks = None # GT binary instance masks with cat-ids as values instead of 1
        self._num_steps = 0
        self._last_actions = {'input_points':[], 'input_labels':[]}
        self._last_reward = 0
        self._last_best_score = 0
        self.render_mode = render_mode

        self.sam_predictor = RepVITSamWrapper(sam_ckpt_fp)

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(0, 255, shape=(*img_shape,), dtype=np.uint8),
                "target_category": spaces.Text(max_length=max(map(len, target_categories))),
                "sam_image_embeddings": spaces.Box(-np.inf, np.inf, shape=(*embedding_shape,), dtype=np.float32),
                "sam_pred_mask_prob": spaces.Box(0, 1, shape=(*mask_shape,), dtype=np.float32),
                "num_steps": spaces.Discrete(max_steps + 1 if max_steps is not None else 1000),
            }
        )

        img_h, img_w = img_shape[:2]
        if img_patch_size is not None:
            width_patch_size = height_patch_size = img_patch_size
            # number of patches along a single dimension is the ceiling of the division of the image size by the patch size
            num_width_patches = img_w//width_patch_size + int(img_w%width_patch_size != 0)
            num_height_patches = img_h//height_patch_size + int(img_h%height_patch_size != 0)
        elif num_patches is not None:
            num_width_patches = num_height_patches = num_patches
            width_patch_size = img_w / num_width_patches
            height_patch_size = img_h / num_height_patches

            self.img_patch_size = max(width_patch_size, height_patch_size) 
        else:
            raise ValueError("Either img_patch_size or num_patches should be provided")
        # print(num_width_patches, num_height_patches)

        # The following dictionary maps abstract actions from `self.action_space` to
        # the input_points and input_labels submitted to SAM in if that action is taken.
        # I.e. 0 corresponds to "(0, 0), positive", 1 to "(0, 0), negative",
        # 2 to "(0, 1), positive", 3 to "(0, 1), negative", etc
        
        self._action_to_input = {}
        for wpatch_idx in range(0, num_width_patches):
            for hpatch_idx in range(0, num_height_patches):
                # Mark patch center as input point (x, y)
                patch_center_x = int(wpatch_idx * width_patch_size + width_patch_size/2)
                patch_center_y = int(hpatch_idx * height_patch_size + height_patch_size/2)
                input_point = (patch_center_x, patch_center_y)
                
                self._action_to_input[len(self._action_to_input)] = (input_point, 'pos')
                self._action_to_input[len(self._action_to_input)] = (input_point, 'neg')

        # Add 2nd last action to remove previous input
        # self._action_to_input[len(self._action_to_input)] = ((1e6,1e6), 'remove')
        # Add last action to mark the task as done
        # self._action_to_input[len(self._action_to_input)] = ((1e6, 1e6), 'done')
        self.action_space = spaces.Discrete(len(self._action_to_input))

        self._load_sample_from_dataset()

    def _apply_augmentation(self, image, mask):
        """Apply data augmentation to image and mask (synchronized transforms)."""
        if not self.augmentation.get('enabled', False):
            return image, mask
        
        aug_cfg = self.augmentation
        h, w = image.shape[:2]
        
        # 1. Geometric transformations (applied to both image and mask)
        center = (w // 2, h // 2)
        
        # Random rotation
        if 'rotate_range' in aug_cfg:
            angle = np.random.uniform(*aug_cfg['rotate_range'])
            M_rot = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, M_rot, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            mask = cv2.warpAffine(mask, M_rot, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
        
        # Random scaling
        if 'scale_range' in aug_cfg:
            scale = np.random.uniform(*aug_cfg['scale_range'])
            M_scale = cv2.getRotationMatrix2D(center, 0, scale)
            image = cv2.warpAffine(image, M_scale, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            mask = cv2.warpAffine(mask, M_scale, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
        
        # Random horizontal flip
        if aug_cfg.get('flip_horizontal', 0) > np.random.rand():
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)
        
        # Random vertical flip
        if aug_cfg.get('flip_vertical', 0) > np.random.rand():
            image = cv2.flip(image, 0)
            mask = cv2.flip(mask, 0)
        
        # Elastic deformation (optional, more expensive)
        if aug_cfg.get('elastic_deform', False):
            alpha = aug_cfg.get('elastic_alpha', 100)
            sigma = aug_cfg.get('elastic_sigma', 10)
            
            # Generate random displacement fields
            dx = gaussian_filter((np.random.rand(h, w) * 2 - 1), sigma) * alpha
            dy = gaussian_filter((np.random.rand(h, w) * 2 - 1), sigma) * alpha
            
            x, y = np.meshgrid(np.arange(w), np.arange(h))
            indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
            
            # Apply to image (all channels)
            for c in range(image.shape[2] if len(image.shape) == 3 else 1):
                channel = image[:, :, c] if len(image.shape) == 3 else image
                image[:, :, c] = map_coordinates(channel, indices, order=1, mode='reflect').reshape(h, w)
            
            # Apply to mask (nearest neighbor for discrete labels)
            if len(mask.shape) == 3:
                for k in range(mask.shape[2]):
                    mask[:, :, k] = map_coordinates(mask[:, :, k], indices, order=0, mode='constant').reshape(h, w)
            else:
                mask = map_coordinates(mask, indices, order=0, mode='constant').reshape(h, w)
        
        # 2. Intensity transformations (image only)
        image = image.astype(np.float32) / 255.0  # Normalize to [0, 1]
        
        # Random brightness
        if 'brightness_range' in aug_cfg:
            factor = np.random.uniform(*aug_cfg['brightness_range'])
            image = np.clip(image * factor, 0, 1)
        
        # Random contrast
        if 'contrast_range' in aug_cfg:
            factor = np.random.uniform(*aug_cfg['contrast_range'])
            mean = image.mean(axis=(0, 1), keepdims=True)
            image = np.clip((image - mean) * factor + mean, 0, 1)
        
        # Random gamma correction
        if 'gamma_range' in aug_cfg:
            gamma = np.random.uniform(*aug_cfg['gamma_range'])
            image = np.power(image, gamma)
        
        # Gaussian noise
        if 'gaussian_noise_std' in aug_cfg:
            noise = np.random.normal(0, aug_cfg['gaussian_noise_std'], image.shape)
            image = np.clip(image + noise, 0, 1)
        
        # Gaussian blur
        if 'gaussian_blur_sigma' in aug_cfg:
            sigma = np.random.uniform(*aug_cfg['gaussian_blur_sigma'])
            if sigma > 0:
                image = gaussian_filter(image, sigma=(sigma, sigma, 0))
        
        image = (image * 255).astype(np.uint8)  # Back to uint8
        
        return image, mask

    def _load_sample_from_dataset(self):
        # Choose a random sample from the target categories
        self._curr_target_cat = np.random.choice(self.target_categories)

        # Expects the image to be in RGB format
        img, categorical_instance_masks = self.dataset.get_sample(
            target_categories=[self._curr_target_cat])

        # Resize first, then augment
        img_resized = cv2.resize(img, self.img_shape[:2][::-1])
        mask_resized = cv2.resize(categorical_instance_masks, 
                                  self.img_shape[:2][::-1], 
                                  interpolation=cv2.INTER_NEAREST)
        
        # Apply data augmentation (anti-overfitting)
        img_aug, mask_aug = self._apply_augmentation(img_resized, mask_resized)
        
        self._image = img_aug
        self.sam_predictor.set_image(self._image)
        self._sam_image_embeddings = self.sam_predictor.get_image_embeddings()
        self._sam_pred_mask_prob = np.ones(self.mask_shape, dtype=np.float32) * 0.5
        self._sam_pred_mask = np.zeros(self.img_shape[:2], dtype=np.float32)
        
        # Ensure the mask has 3 dimensions (H, W, K)
        if len(mask_aug.shape) == 2:
            mask_aug = mask_aug[..., np.newaxis]
        self._categorical_instance_masks = mask_aug


    def _get_obs(self):
        return {
            "image": self._image, 
            "target_category": self._curr_target_cat,
            "sam_image_embeddings": self._sam_image_embeddings,
            "sam_pred_mask_prob": self._sam_pred_mask_prob,
            "num_steps": self._num_steps,
        }
    

    def _get_info(self):
        # Convert categorical instance masks to binary ground truth for Phase 2
        if self._categorical_instance_masks is not None:
            gt_mask_binary = (np.any(self._categorical_instance_masks > 0, axis=-1) if len(self._categorical_instance_masks.shape) == 3 
                             else self._categorical_instance_masks > 0).astype(np.float32)
        else:
            gt_mask_binary = None

        # IoU for validation / early stopping (final mask vs GT)
        iou = None
        if gt_mask_binary is not None and self._sam_pred_mask is not None:
            resized_gt = cv2.resize(
                gt_mask_binary.astype(np.float32),
                self._sam_pred_mask.shape[::-1],
                interpolation=cv2.INTER_NEAREST,
            )
            pred_binary = (self._sam_pred_mask >= 0.5).astype(np.float32)
            inter = np.sum(resized_gt * pred_binary)
            union = np.sum(resized_gt) + np.sum(pred_binary) - inter
            eps = 1e-6
            iou = float((inter / (union + eps)))

        return {
            "last_input_points": self._last_actions["input_points"].copy(),
            "last_input_labels": self._last_actions["input_labels"].copy(),
            "sam_pred_mask": self._sam_pred_mask.copy(),
            "dice_score": self._last_best_score,
            "iou": iou,
            "ground_truth_mask": gt_mask_binary,  # Added for Phase 2
            "current_mask": self._sam_pred_mask.copy(),  # Alias for clarity
        }
    

    def run_sam(self):
        masks, ious, low_res_mask_logits = self.sam_predictor.predict(
            np.array(self._last_actions["input_points"]), 
            np.array(self._last_actions["input_labels"])
        )

        best_mask_idx = np.argmax(ious)
        # Use numerically stable sigmoid to prevent overflow
        logits_clipped = np.clip(low_res_mask_logits[best_mask_idx], -500, 500)
        best_pred_mask_prob = 1 / (1 + np.exp(-logits_clipped))
        
        # Resize mask_prob to mask_shape if needed
        if best_pred_mask_prob.shape != self.mask_shape:
            best_pred_mask_prob = cv2.resize(best_pred_mask_prob, self.mask_shape[::-1], interpolation=cv2.INTER_LINEAR)
        self._sam_pred_mask_prob = best_pred_mask_prob
        
        # Also fix for mask computation - masks from SAM are in image resolution
        mask_logits_clipped = np.clip(masks[best_mask_idx], -500, 500)
        mask_prob = (1 / (1 + np.exp(-mask_logits_clipped))).astype(np.float32)
        
        # Resize mask to img_shape if needed (SAM returns masks in original image size)
        if mask_prob.shape != self.img_shape[:2]:
            mask_prob = cv2.resize(mask_prob, self.img_shape[:2][::-1], interpolation=cv2.INTER_LINEAR)
        
        self._sam_pred_mask = mask_prob
        
        # Debug: Print mask info for first few calls
        if not hasattr(self, '_debug_run_sam_count'):
            self._debug_run_sam_count = 0
        self._debug_run_sam_count += 1
        if self._debug_run_sam_count <= 3:
            print(f"[ENV DEBUG] run_sam: mask shape={self._sam_pred_mask.shape}, sum={self._sam_pred_mask.sum():.2f}, max={self._sam_pred_mask.max():.4f}")


    def convert_raw_input_to_action(self, input_point, input_label):
        point_dist = lambda x: np.linalg.norm([input_point[0]-x[0], input_point[1]-x[1]])
        input_dist = np.array([point_dist(point) + (1e6 * (label != input_label)) 
                               for (point, label) in self._action_to_input.values() 
                               if type(point) == tuple])
        sample_action = np.argmin(input_dist)

        return sample_action
    

    def refine_input_point(self, input_point, input_type):
        if input_type != 'pos':
            return input_point
        
        if self._categorical_instance_masks is None:
            return input_point
        
        if np.any(self._categorical_instance_masks[input_point[1], input_point[0]] == 1):
            return input_point
        
        # Move to match the center of foreground portion of the patch, if present
        patch_half_size = int(self.img_patch_size // 2)
        patch_around_point_xrange = (max(0, input_point[0] - patch_half_size),
                                        min(self.img_shape[1], input_point[0] + patch_half_size))
        patch_around_point_yrange = (max(0, input_point[1] - patch_half_size),
                                        min(self.img_shape[0], input_point[1] + patch_half_size))
        # Get the patch around the point
        mask_patch = self._categorical_instance_masks[patch_around_point_yrange[0]:patch_around_point_yrange[1],
                                                        patch_around_point_xrange[0]:patch_around_point_xrange[1]]
        mask_patch = np.any(mask_patch > 0, axis=-1)  # Combine all instance masks into a single binary mask    

        # If the patch is empty, return the original action
        if not np.any(mask_patch > 0):
            return input_point
        
        # Move the input point to the center of the foreground portion of the patch
        foreground_indices = np.argwhere(mask_patch)
        foreground_indices_center_x = np.mean(foreground_indices[:, 1]) + patch_around_point_xrange[0]
        foreground_indices_center_y = np.mean(foreground_indices[:, 0]) + patch_around_point_yrange[0]
        input_point = (int(foreground_indices_center_x), int(foreground_indices_center_y)) 
        
        return input_point

    def _compute_terminal_reward(self):
        """Terminal acceptability: r_T += (2*y_T - 1), y_T = 1[Dice(M_T,G) >= tau_acc]."""
        resized_gt_mask = cv2.resize(self._categorical_instance_masks,
                                     self._sam_pred_mask.shape[::-1],
                                     interpolation=cv2.INTER_NEAREST)
        if len(resized_gt_mask.shape) == 2:
            resized_gt_mask = resized_gt_mask[..., np.newaxis]
        gt_mask = np.any(resized_gt_mask > 0, axis=-1).astype(np.float32)
        eps = 1e-6
        intersection = np.sum(gt_mask * self._sam_pred_mask)
        union = np.sum(gt_mask) + np.sum(self._sam_pred_mask)
        dice = (2 * intersection + eps) / (union + eps)
        y_T = 1 if dice >= self.tau_acc else 0
        return 2 * y_T - 1

    def compute_reward(self, pred_mask, act):
        # Compute dice score
        resized_gt_mask = cv2.resize(self._categorical_instance_masks,
                                     pred_mask.shape[::-1],
                                     interpolation=cv2.INTER_NEAREST)
        # Ensure the resized_gt_mask has 3 dimensions (H, W, K)
        if len(resized_gt_mask.shape) == 2:
            resized_gt_mask = resized_gt_mask[..., np.newaxis]

        if self.use_dice_score:
            eps = 1e-6
            # Flatten the binary masks per instance to compute the dice score
            gt_mask = np.any(resized_gt_mask > 0, axis=-1).astype(np.float32)
            intersection = np.sum(gt_mask * pred_mask)
            union = np.sum(gt_mask) + np.sum(pred_mask) #- intersection
            dice_score = (2 * intersection + eps)/ (union + eps)

            dice_gain = max(0, dice_score - self._last_best_score)
            self._last_best_score = max(dice_score, self._last_best_score)
        else:
            dice_score = 0.0
            dice_gain = 0.0

        # r_t = d_coeff * r_correct(s_t,a_t) + r_quality(s_t,s_{t+1}) for t < T
        r_correct = 0.0
        dist_coeff = 1.0
        if act == 'add':
            input_point = self._last_actions["input_points"][-1]
            input_label = self._last_actions["input_labels"][-1]

            # d_coeff: normalized distance to nearest previously placed prompt (same type)
            if len(self._last_actions["input_points"]) > 1:
                point_dist = lambda x: np.linalg.norm([input_point[0]-x[0], input_point[1]-x[1]])
                input_dist = np.array([point_dist(point) + (1e6 * (label != input_label)) \
                                       for (point, label) in zip(self._last_actions["input_points"][:-1], 
                                                                 self._last_actions["input_labels"][:-1])])
                dist_coeff = 1 - np.exp(-np.min(input_dist) / (2 * self.img_patch_size))
            else:
                dist_coeff = 1.0

            point_image_indices = tuple(map(int, (input_point[1], input_point[0])))
            gt_label = np.any(self._categorical_instance_masks[point_image_indices] > 0)

            if gt_label == 1:
                r_correct = float(input_label == gt_label)
            if self.penalize_for_wrong_input and (input_label != gt_label):
                r_correct = -0.5 if gt_label == 1 else -0.1

        # r_quality: quality-improving refinements (dice gain)
        r_quality = float(dice_gain) if self.use_dice_score else 0.0

        reward = dist_coeff * r_correct + r_quality

        self._last_reward = reward

        return reward


    def step(self, action):
        # Debug: Check action type and value (only for first few steps)
        if not hasattr(self, '_debug_step_count'):
            self._debug_step_count = 0
        self._debug_step_count += 1
        
        # Handle VectorEnv wrapping - action might be an array
        action_original = action
        if isinstance(action, (np.ndarray, list, tuple)):
            if len(action) > 0:
                action = action[0] if isinstance(action, np.ndarray) else action[0]
            else:
                raise ValueError(f"Empty action array: {action_original}")
        # Convert to int if it's a numpy scalar
        if isinstance(action, (np.integer, np.floating)):
            action = int(action)
        
        if action < 0 or action >= self.action_space.n:
            raise ValueError(f"Invalid action: {action} (type={type(action)}, action_space.n={self.action_space.n}, original={action_original})")
        
        input_point, input_type = self._action_to_input[action]
        
        # Debug: Print first few actions
        if self._debug_step_count <= 3:
            print(f"[ENV DEBUG] Step {self._debug_step_count}: action={action}, input_point={input_point}, input_type={input_type}")
        
        terminated = input_type == 'done'
        if terminated:
            # r_T += (2*y_T - 1), y_T = 1[Dice(M_T,G) >= tau_acc]
            terminal_reward = self._compute_terminal_reward()
            return self._get_obs(), terminal_reward, terminated, False, self._get_info()
        
    
        act = None
        if input_type == 'remove':
            # Remove the last input
            if len(self._last_actions["input_points"]) > 0:
                self._last_actions["input_points"].pop()
                self._last_actions["input_labels"].pop()
                act = 'remove'
        elif input_type in ['pos', 'neg']:
            # Add a new input point
            input_label = 1 if input_type == 'pos' else 0 # label 0-neg, 1-pos

            # Refine the input to match the center of the 
            # foreground portion of the patch, if present
            input_point = self.refine_input_point(input_point, input_type)

            self._last_actions["input_points"].append(input_point)
            self._last_actions["input_labels"].append(input_label)
            act = 'add'
            
            # Debug: Print after adding point
            if self._debug_step_count <= 3:
                print(f"[ENV DEBUG] Added point: {input_point}, label={input_label}, total_points={len(self._last_actions['input_points'])}")
        elif input_type == 'next':
            #TODO : Not implemented yet, 
            # Is meant to be used for predicting the next instance mask
            act = 'next'
            self._last_actions = {'input_points':[], 'input_labels':[]}
        else:
            raise ValueError(f"Invalid input type: {input_type}")

        if len(self._last_actions["input_points"]) > 0:
            self.run_sam()
        else:
            # Set the mask and mask_prob to initial state
            self._sam_pred_mask_prob = np.ones(self.mask_shape, dtype=np.float32) * 0.5
            self._sam_pred_mask = np.zeros(self.img_shape[:2], dtype=np.float32)
        
        self._num_steps += 1

        reward = self.compute_reward(self._sam_pred_mask, act)

        trunc = (self.max_steps is not None and self._num_steps >= self.max_steps)
        if trunc:
            # Terminal step: add r_T += (2*y_T - 1)
            reward += self._compute_terminal_reward()

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, False, trunc, info
    

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        self._load_sample_from_dataset()

        self._last_actions = {'input_points':[], 'input_labels':[]}
        self._num_steps = 0
        self._last_reward = 0
        self._last_best_score = 0

        observation = self._get_obs()
        info = self._get_info()

        return observation, info
    

    def render(self):
        img = cv2.resize(cv2.cvtColor(self._image, cv2.COLOR_RGB2BGR), self.render_frame_shape[::-1])

        mask_instances_color = np.zeros((*self._categorical_instance_masks.shape[:2], 3), dtype=np.uint8)
        num_instances = self._categorical_instance_masks.shape[-1]
        for idx in range(num_instances):
            instance_color = (idx + 1) * int(255 / num_instances)
            mask_instances_color[self._categorical_instance_masks[..., idx] > 0] = [
                instance_color, instance_color, instance_color
            ]
        gt_mask = cv2.resize(mask_instances_color, self.render_frame_shape[::-1])


        mask = cv2.cvtColor((self._sam_pred_mask * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        for point,label in zip(self._last_actions["input_points"], self._last_actions["input_labels"]):
            cv2.circle(mask, point, 10, (0, 255, 0) if label == 1 else (0, 0, 255), -1)
        mask = cv2.resize(mask, self.render_frame_shape[::-1])
        
        concat_img = np.concatenate([img, gt_mask, mask], axis=1)

        if self.render_mode == 'rgb_array':
            concat_img = cv2.cvtColor(concat_img, cv2.COLOR_BGR2RGB)

        # Add the target category text to the image
        cv2.putText(concat_img,
                    self._curr_target_cat,
                    (10, concat_img.shape[0]//2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2)
        
        # Add the last reward text to the image
        cv2.putText(concat_img,
                    f"Reward: {self._last_reward:.2f}",
                    (10, concat_img.shape[0]//2 + 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2)
        return concat_img


if __name__ == "__main__":
    import os

    img_shape = (375, 500, 3)  # HxWxC
    embedding_shape = (256, 64, 64)  # CxHxW
    mask_shape = (256, 256)  # HxW
    render_frame_shape = (320, 426)  # HxW
    max_steps = 5
    penalize_for_wrong_input = True
    use_dice_score = True
    img_patch_size = 32
    render_mode = 'human'

    target_categories = ['colon']
    dataset_config = {
        'type': 'medical',
        'data_dir': os.path.join(os.getcwd(), 'images', 'Task10_Colon'),
        'split_pkl_path': os.path.join(os.getcwd(), 'images', 'Task10_Colon', 'split.pkl'),
        'split_idx': 0,
        'data_type': 'val',
        'seed': 42,
        'max_instances': 5,
    }

    sam_ckpt_fp = os.path.join(os.getcwd(), 'RepViT', 'sam', 'weights', 'repvit_sam.pt')

    env = SamSegEnv(img_shape=img_shape, 
                    embedding_shape=embedding_shape,
                    mask_shape=mask_shape,
                    render_frame_shape=render_frame_shape,
                    max_steps=max_steps,
                    target_categories=target_categories,
                    dataset_config=dataset_config,
                    penalize_for_wrong_input=penalize_for_wrong_input,
                    use_dice_score=use_dice_score,
                    sam_ckpt_fp=sam_ckpt_fp,
                    img_patch_size=img_patch_size,
                    render_mode=render_mode)

    print(env.action_space.n)
    obs, info = env.reset()
    
    global sample_action
    sample_action = env.action_space.n - 2 #np.random.randint(0, env.action_space.n)

    def get_action(event,x,y,flags,param):
        global sample_action
        if event == cv2.EVENT_LBUTTONUP:
            tgt_label = 'pos'
        elif event == cv2.EVENT_RBUTTONUP:
            tgt_label = 'neg'
        else:
            return
        # Ensure the coordinates are within the image bounds
        # Since we concat the images horizontally, we need to adjust the coordinates
        x = x % render_frame_shape[1]
        y = y % render_frame_shape[0]

        # Scale the coordinates to match the original image size
        scaled_x = int(x * img_shape[1] / render_frame_shape[1])
        scaled_y = int(y * img_shape[0] / render_frame_shape[0])

        # Convert the raw input to an action
        sample_action = env.convert_raw_input_to_action((scaled_x, scaled_y), tgt_label)


    cv2.namedWindow('image', cv2.WINDOW_GUI_NORMAL)
    cv2.setMouseCallback('image', get_action)
    total_reward = 0
    while True:
        obs, reward, done, trunc, info = env.step(sample_action)
        print('reward:', reward)
        total_reward += reward

        # print(obs['image'].shape, info['sam_pred_mask'].shape)
        print(info['last_input_labels'], info['last_input_points'])

        if done or trunc:
            print("Task done, Total reward:", total_reward)
            print("-"*50)
            total_reward = 0
            obs, info = env.reset()

        frame = env.render()
        cv2.imshow('image', frame)

        sample_action = env.action_space.n - 1  # Mark task as done by default

        if cv2.waitKey(0) == 27:  # Esc key to stop the loop
            break


    cv2.destroyAllWindows()



    