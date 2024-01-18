from typing import Dict, Sequence, TypedDict

import torch


# define a new type of data, name -> track audio 
AudioDict = Dict[str, torch.Tensor] 


DataDict = TypedDict(
    'DataDict', 
    {'audio': AudioDict, 'track': str})

BatchedDataDict = TypedDict(
        'BatchedDataDict',
        {'audio': AudioDict, 'track': Sequence[str]}
)


#new type of data
class DataDictWithLanguage(TypedDict):
    audio: AudioDict
    track: str
    language: str