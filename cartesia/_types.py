from enum import Enum
from typing import List, Optional, TypedDict, Union

try:
    import numpy as np

    _NUMPY_AVAILABLE = True
except ImportError:
    _NUMPY_AVAILABLE = False


class AudioDataReturnType(Enum):
    BYTES = "bytes"
    ARRAY = "array"


class AudioOutputFormat(Enum):
    """Supported output formats for the audio."""

    FP32 = "fp32"  # float32
    PCM = "pcm"  # 16-bit signed integer PCM
    FP32_16000 = "fp32_16000"  # float32, 16 kHz
    FP32_22050 = "fp32_22050"  # float32, 22.05 kHz
    FP32_44100 = "fp32_44100"  # float32, 44.1 kHz
    PCM_16000 = "pcm_16000"  # 16-bit signed integer PCM, 16 kHz
    PCM_22050 = "pcm_22050"  # 16-bit signed integer PCM, 22.05 kHz
    PCM_44100 = "pcm_44100"  # 16-bit signed integer PCM, 44.1 kHz


class AudioOutput(TypedDict):
    audio: Union[bytes, "np.ndarray"]
    sampling_rate: int


Embedding = List[float]


class VoiceMetadata(TypedDict):
    id: str
    name: str
    description: str
    embedding: Optional[Embedding]
