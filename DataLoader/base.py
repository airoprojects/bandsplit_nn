import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional 

import pedalboard as pb # library developed by Spotify to add effect on audio etc...
import numpy as np

import torch
import torchaudio as ta
from torch.utils import data

from datatype import AudioDict, DataDict


# Define a base class for source separation datasets, inheriting from PyTorch's data.Dataset and Python's ABC for abstract classes
class BaseSourceSeparationDataset(data.Dataset, ABC):
    def __init__(
            self, split: str,
            stems: List[str],
            files: List[str],
            data_path: str,
            fs: int,
            npy_memmap: bool,
            recompute_mixture: bool
            ):
        self.split = split
        self.stems = stems
        self.stems_no_mixture = [s for s in stems if s != "mixture"]
        self.files = files
        self.data_path = data_path
        self.fs = fs
        self.npy_memmap = npy_memmap
        self.recompute_mixture = recompute_mixture

    @abstractmethod
    def get_stem(
            self,
            *,
            stem: str,
            identifier: Dict[str, Any]
            ) -> torch.Tensor:
        raise NotImplementedError

    def _get_audio(self, stems, identifier: Dict[str, Any]):
        audio = {}
        for stem in stems:
            audio[stem] = self.get_stem(stem=stem, identifier=identifier)

        return audio

    def get_audio(self, identifier: Dict[str, Any]) -> AudioDict:

        if self.recompute_mixture:
            audio = self._get_audio(
                self.stems_no_mixture,
                identifier=identifier
                )
            audio["mixture"] = self.compute_mixture(audio)
            return audio
        else:
            return self._get_audio(self.stems, identifier=identifier)

    @abstractmethod
    def get_identifier(self, index: int) -> Dict[str, Any]:
        pass

    def compute_mixture(self, audio: AudioDict) -> torch.Tensor:

        return sum(
                audio[stem] for stem in audio if stem != "mixture"
        )
    









''' the versione commentend of the code


# Import necessary libraries and modules
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np
import pedalboard as pb 
import torch
import torchaudio as ta
from torch.utils import data

# Import custom types defined elsewhere in the project
from datatype import AudioDict, DataDict


# Define a base class for source separation datasets, inheriting from PyTorch's data.Dataset and Python's ABC for abstract classes
class BaseSourceSeparationDataset(data.Dataset, ABC):
    def __init__(
            self, split: str,
            stems: List[str],
            files: List[str],
            data_path: str,
            fs: int,
            npy_memmap: bool,
            recompute_mixture: bool
            ):
        # Initializing the dataset with various parameters
        self.split = split  # Dataset split (e.g., 'train', 'test')
        self.stems = stems  # List of stems (individual audio tracks) in the dataset
        self.stems_no_mixture = [s for s in stems if s != "mixture"]  # Stems excluding the 'mixture'
        self.files = files  # List of file names in the dataset
        self.data_path = data_path  # Path to the data
        self.fs = fs  # Sampling frequency
        self.npy_memmap = npy_memmap  # Boolean indicating if memory mapping is used for numpy arrays
        self.recompute_mixture = recompute_mixture  # Boolean indicating if mixtures should be recomputed

    @abstractmethod
    def get_stem(
            self,
            *,
            stem: str,
            identifier: Dict[str, Any]
            ) -> torch.Tensor:
        # Abstract method that needs to be implemented in a subclass; it defines how to get an individual stem's audio data
        raise NotImplementedError

    def _get_audio(self, stems, identifier: Dict[str, Any]):
        # Private method to retrieve audio data for the specified stems
        audio = {}
        for stem in stems:
            # Fetching each stem's audio data using the abstract 'get_stem' method
            audio[stem] = self.get_stem(stem=stem, identifier=identifier)

        return audio

    def get_audio(self, identifier: Dict[str, Any]) -> AudioDict:
        # Public method to get audio data; handles the logic of whether to recompute the mixture or not

        if self.recompute_mixture:
            # If recompute_mixture is True, compute the audio without the mixture and then add the mixture
            audio = self._get_audio(
                self.stems_no_mixture,
                identifier=identifier
                )
            # Compute the mixture audio track and add it to the audio dictionary
            audio["mixture"] = self.compute_mixture(audio)
            return audio
        else:
            # If recompute_mixture is False, get the audio for all stems
            return self._get_audio(self.stems, identifier=identifier)

    @abstractmethod
    def get_identifier(self, index: int) -> Dict[str, Any]:
        # Abstract method to retrieve an identifier for a given dataset index; needs implementation in subclasses
        pass

    def compute_mixture(self, audio: AudioDict) -> torch.Tensor:
        # Method to compute the mixture audio track by summing the individual stems' audio, excluding the 'mixture' stem itself
        return sum(
                audio[stem] for stem in audio if stem != "mixture"
        )


        
'''