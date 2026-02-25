from datasets.medical_dataset import MedicalDataset


def get_dataset(dataset_config):
    """
    Factory function to get the dataset based on the configuration.
    Phase I uses the medical dataset only.
    """
    if dataset_config["type"] != "medical":
        raise ValueError(f"Unsupported dataset type: {dataset_config['type']}. Only 'medical' is supported.")
    return MedicalDataset(**dataset_config)
