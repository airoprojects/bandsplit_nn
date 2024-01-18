#os 
import os


#numpy
import numpy as np

#torch import 
import torch 
import torchaudio as ta
from torch.utils import data

#data type import 
from datatype import AudioDict, DataDict

from base import BaseSourceSeparationDataset



class DivideAndRemasterBaseDataset():
  


'''
commented version

# Base class for the Divide and Remaster dataset, inheriting from a custom source separation dataset class
class DivideAndRemasterBaseDataset(BaseSourceSeparationDataset, ABC):
    # Allowed stems (audio tracks) and naming conventions for different dataset splits
    ALLOWED_STEMS = ["mixture", "speech", "music", "effects", "mne"]
    STEM_NAME_MAP = {"mixture": "mix", "speech": "speech", "music": "music", "effects": "sfx"}
    SPLIT_NAME_MAP = {"train": "tr", "val": "cv", "test": "tt"}

    # Constants for audio track length in seconds and samples
    FULL_TRACK_LENGTH_SECOND = 60
    FULL_TRACK_LENGTH_SAMPLES = FULL_TRACK_LENGTH_SECOND * 44100

    # Constructor for initializing the dataset
    def __init__(self, split: str, stems: List[str], files: List[str], data_path: str, fs: int = 44100, npy_memmap: bool = True, recompute_mixture: bool = False) -> None:
        super().__init__(split=split, stems=stems, files=files, data_path=data_path, fs=fs, npy_memmap=npy_memmap, recompute_mixture=recompute_mixture)

    # Method to retrieve a specific audio stem from the dataset
    def get_stem(self, *, stem: str, identifier: Dict[str, Any]) -> torch.Tensor:
        # Special handling for the 'mne' stem (music + effects)
        if stem == "mne":
            return self.get_stem(stem="music", identifier=identifier) + self.get_stem(stem="effects", identifier=identifier)

        # Loading the audio file from the dataset
        track = identifier["track"]
        path = os.path.join(self.data_path, track)
        # Load audio using numpy memmap or torchaudio based on npy_memmap flag
        # ...

    # Method to get the identifier for a dataset item
    def get_identifier(self, index):
        return dict(track=self.files[index])

    # Method to get an item from the dataset
    def __getitem__(self, index: int) -> DataDict:
        identifier = self.get_identifier(index)
        audio = self.get_audio(identifier)
        return {"audio": audio, "track": f"{self.split}/{identifier['track']}"}

# Class for the standard Divide and Remaster Dataset
class DivideAndRemasterDataset(DivideAndRemasterBaseDataset):
    # Constructor to initialize the dataset with specific parameters
    def __init__(self, data_root: str, split: str, stems: Optional[List[str]] = None, fs: int = 44100, npy_memmap: bool = True) -> None:
        # Default stems if not provided and construction of the data path
        # Listing and filtering files in the dataset directory
        # Initializing the base class with the dataset parameters
        # ...

    # Method to get the number of items in the dataset
    def __len__(self) -> int:
        return self.n_tracks

# Class for handling random chunks of the Divide and Remaster Dataset
class DivideAndRemasterRandomChunkDataset(DivideAndRemasterBaseDataset):
    # Constructor with additional parameters for handling chunks
    def __init__(self, data_root: str, split: str, target_length: int, chunk_size_second: float, stems: Optional[List[str]] = None, fs: int = 44100, npy_memmap: bool = True) -> None:
        # Initialization similar to DivideAndRemasterDataset with additional logic for chunk handling
        # ...

    # Overridden methods for dataset length, item retrieval, and stem retrieval with chunking logic

# Class for handling deterministic chunks of the Divide and Remaster Dataset
class DivideAndRemasterDeterministicChunkDataset(DivideAndRemasterBaseDataset):
    # Constructor with parameters for deterministic chunking
    def __init__(self, data_root: str, split: str, chunk_size_second: float, hop_size_second: float, stems: Optional[List[str]] = None, fs: int = 44100, npy_memmap: bool = True) -> None:
        # Similar initialization to the random chunk dataset but with deterministic chunking
        # ...

    # Overridden methods for dataset length and item retrieval with deterministic chunking

# Class for handling random chunks of the dataset with additional speech reverb processing
class DivideAndRemasterRandomChunkDatasetWithSpeechReverb(DivideAndRemasterRandomChunkDataset):
    # Constructor similar to the random chunk dataset
    def __init__(self, data_root: str, split: str, target_length: int, chunk_size_second: float, stems: Optional[List[str]] = None, fs: int = 44100, npy_memmap: bool = True) -> None:
        # Initialization with handling for stems without the mixture
        # ...

    # Overridden method for item retrieval with additional speech reverb processing
    def __getitem__(self, index: int) -> DataDict:
        # Process of applying reverb to speech and adjusting the mixture accordingly
        # ...

    # Method to get the length of the dataset

# Main execution block for testing the datasets
if __name__ == "__main__":
    # Looping through different dataset splits and printing their length
    # Iterating through the dataset and printing the shape of audio tracks
    # ...
'''

