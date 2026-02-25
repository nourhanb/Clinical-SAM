import os
import pickle
import cv2
import numpy as np
import nibabel as nib
from skimage import exposure


class MedicalDataset:
    """
    Dataset loader for medical imaging data in NIfTI format.
    Supports 3D volumes and extracts 2D slices for training.
    """

    def __init__(self, data_dir, split_pkl_path, split_idx=0, data_type='train', 
                 seed=32, max_instances=-1, slice_axis=2, min_slice_area=0.01, 
                 cache_volumes=False, max_cache_size=2, **kwargs):
        """
        Args:
            data_dir: Root directory containing the dataset (e.g., 'images/Task10_Colon')
            split_pkl_path: Path to the split.pkl file
            split_idx: Index of the split to use (default: 0)
            data_type: 'train', 'val', or 'test'
            seed: Random seed for reproducibility
            max_instances: Maximum number of instances to return per sample
            slice_axis: Axis along which to extract slices (0=sagittal, 1=coronal, 2=axial)
            min_slice_area: Minimum area fraction for a slice to be considered valid
            cache_volumes: Whether to cache volumes in memory (default: False to save memory)
            max_cache_size: Maximum number of volumes to cache if caching is enabled
        """
        super().__init__()
        self.data_dir = data_dir
        self.data_type = data_type
        self.split_idx = split_idx
        self.slice_axis = slice_axis
        self.min_slice_area = min_slice_area
        self.random = np.random.RandomState(seed=seed)
        self.max_instances = max_instances if max_instances > 0 else 100
        self.cache_volumes = cache_volumes
        self.max_cache_size = max_cache_size
        
        # Load split information
        with open(split_pkl_path, 'rb') as f:
            splits = pickle.load(f)
        
        if split_idx >= len(splits):
            raise ValueError(f"split_idx {split_idx} is out of range. Available splits: {len(splits)}")
        
        split_data = splits[split_idx]
        if data_type not in split_data:
            raise ValueError(f"data_type '{data_type}' not found in split. Available: {split_data.keys()}")
        
        self.cases = split_data[data_type]
        self.case_names = list(self.cases.keys())
        
        # Cache for loaded volumes (to avoid reloading) - disabled by default to save memory
        self._volume_cache = {}
        self._mask_cache = {}
        self._cache_access_order = []  # For LRU cache eviction

    def _load_nifti(self, file_path):
        """Load a NIfTI file and return the data array."""
        # Check cache if caching is enabled
        if self.cache_volumes and file_path in self._volume_cache:
            # Update access order for LRU
            if file_path in self._cache_access_order:
                self._cache_access_order.remove(file_path)
            self._cache_access_order.append(file_path)
            return self._volume_cache[file_path]
        
        full_path = os.path.join(self.data_dir, file_path)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"NIfTI file not found: {full_path}")
        
        nii = nib.load(full_path)
        data = nii.get_fdata()
        
        # Cache if enabled and cache not full
        if self.cache_volumes:
            # Evict oldest if cache is full
            if len(self._volume_cache) >= self.max_cache_size:
                oldest = self._cache_access_order.pop(0)
                del self._volume_cache[oldest]
                if oldest in self._mask_cache:
                    del self._mask_cache[oldest]
            
            self._volume_cache[file_path] = data
            self._cache_access_order.append(file_path)
        
        return data

    def _normalize_to_uint8(self, img_slice):
        """Normalize a 2D slice to uint8 range [0, 255]."""
        # Handle different data types
        if img_slice.dtype == np.uint8:
            return img_slice
        
        # Normalize to [0, 1] then scale to [0, 255]
        img_min = img_slice.min()
        img_max = img_slice.max()
        
        if img_max > img_min:
            img_normalized = (img_slice - img_min) / (img_max - img_min)
        else:
            img_normalized = np.zeros_like(img_slice)
        
        # Apply contrast enhancement (optional, helps with medical images)
        img_normalized = exposure.rescale_intensity(img_normalized, out_range=(0, 1))
        
        return (img_normalized * 255).astype(np.uint8)

    def _slice_to_rgb(self, img_slice):
        """Convert a 2D grayscale slice to RGB format."""
        img_uint8 = self._normalize_to_uint8(img_slice)
        # Stack to create RGB (grayscale image repeated 3 times)
        return np.stack([img_uint8, img_uint8, img_uint8], axis=-1)

    def _extract_slice(self, volume, slice_idx, axis):
        """Extract a 2D slice from a 3D volume along the specified axis."""
        if axis == 0:
            return volume[slice_idx, :, :]
        elif axis == 1:
            return volume[:, slice_idx, :]
        elif axis == 2:
            return volume[:, :, slice_idx]
        else:
            raise ValueError(f"Invalid axis: {axis}. Must be 0, 1, or 2.")

    def _get_valid_slices(self, mask_volume, axis):
        """Get indices of slices that contain valid masks."""
        valid_slices = []
        depth = mask_volume.shape[axis]
        
        for i in range(depth):
            if axis == 0:
                mask_slice = mask_volume[i, :, :]
            elif axis == 1:
                mask_slice = mask_volume[:, i, :]
            else:
                mask_slice = mask_volume[:, :, i]
            
            # Check if slice has any non-zero pixels
            if np.any(mask_slice > 0):
                valid_slices.append(i)
        
        return valid_slices

    def _get_instances_from_mask(self, mask_slice):
        """Extract individual instances from a mask slice.
        Returns a list of binary masks, one per instance.
        """
        masks = []
        unique_labels = np.unique(mask_slice)
        unique_labels = unique_labels[unique_labels > 0]  # Remove background
        
        for label in unique_labels:
            instance_mask = (mask_slice == label).astype(np.uint8)
            masks.append(instance_mask)
        
        return masks

    def load_image(self, case_name, slice_idx=None):
        """
        Load an image slice from a case.
        If slice_idx is None, a random valid slice is selected.
        """
        if case_name not in self.cases:
            raise ValueError(f"Case '{case_name}' not found in {self.data_type} split")
        
        image_path = self.cases[case_name][0]
        volume = self._load_nifti(image_path)
        
        # If slice_idx not provided, select a random slice
        if slice_idx is None:
            depth = volume.shape[self.slice_axis]
            slice_idx = self.random.randint(0, depth)
        
        img_slice = self._extract_slice(volume, slice_idx, self.slice_axis)
        img_rgb = self._slice_to_rgb(img_slice)
        
        return img_rgb, slice_idx

    def get_sample(self, target_categories):
        """
        Get a random sample from the dataset.
        
        Args:
            target_categories: List of target category names (for medical data, 
                              typically just ['colon'] or similar). This parameter
                              is kept for compatibility with COCO interface but
                              is not strictly used for filtering in medical datasets.
        
        Returns:
            img: RGB image (H, W, 3) as numpy array
            categorical_instance_masks: Stacked masks (H, W, K) where K is number of instances
        """
        # For medical datasets, we typically have one category
        # The target_categories parameter is kept for interface compatibility
        if target_categories is None or len(target_categories) == 0:
            target_categories = ['colon']  # Default category name
        
        while True:
            # Select a random case
            case_name = self.random.choice(self.case_names)
            
            # Load image and mask volumes
            image_path, label_path = self.cases[case_name]
            img_volume = self._load_nifti(image_path)
            
            # Load mask volume (with caching if enabled)
            if self.cache_volumes and label_path in self._mask_cache:
                mask_volume = self._mask_cache[label_path]
            else:
                mask_volume = self._load_nifti(label_path)
                if self.cache_volumes:
                    self._mask_cache[label_path] = mask_volume
            
            # Get valid slices (slices that contain masks)
            valid_slices = self._get_valid_slices(mask_volume, self.slice_axis)
            
            if len(valid_slices) == 0:
                continue
            
            # Select a random valid slice
            slice_idx = self.random.choice(valid_slices)
            
            # Extract image slice
            img_slice = self._extract_slice(img_volume, slice_idx, self.slice_axis)
            img_rgb = self._slice_to_rgb(img_slice)
            h, w = img_rgb.shape[:2]
            
            # Extract mask slice
            mask_slice = self._extract_slice(mask_volume, slice_idx, self.slice_axis)
            
            # Get instances from mask
            instance_masks = self._get_instances_from_mask(mask_slice)
            
            if len(instance_masks) == 0:
                continue
            
            # Filter by area and limit to max_instances
            filtered_masks = []
            for mask in instance_masks:
                area = np.sum(mask > 0)
                if area >= self.min_slice_area * h * w:
                    filtered_masks.append(mask)
            
            if len(filtered_masks) == 0:
                continue
            
            # Sort by area (largest first) and take top max_instances
            filtered_masks.sort(key=lambda m: np.sum(m > 0), reverse=True)
            filtered_masks = filtered_masks[:self.max_instances]
            
            # Stack masks and assign categorical values (1, 2, 3, ...)
            # Each mask channel gets a unique categorical value
            sample_mask = np.zeros((h, w, len(filtered_masks)), dtype=np.float32)
            for idx, mask in enumerate(filtered_masks):
                sample_mask[..., idx] = mask * (idx + 1)  # Assign categorical value
            
            break
        
        return img_rgb, sample_mask

    def configure_targets(self, target_categories):
        """
        Configure target categories. For medical datasets, this is mainly for compatibility.
        Returns the target categories and available case names.
        """
        return target_categories, self.case_names

