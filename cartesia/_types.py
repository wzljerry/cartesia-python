from typing import List, TypedDict

class OutputFormatMapping:
    _format_mapping = {
        "fp32": {"container": "raw", "encoding": "pcm_f32le", "sample_rate": 44100},
        "pcm": {"container": "raw", "encoding": "pcm_s16le", "sample_rate": 44100},
        "fp32_16000": {"container": "raw", "encoding": "pcm_f32le", "sample_rate": 16000},
        "fp32_22050": {"container": "raw", "encoding": "pcm_f32le", "sample_rate": 22050},
        "fp32_44100": {"container": "raw", "encoding": "pcm_f32le", "sample_rate": 44100},
        "pcm_16000": {"container": "raw", "encoding": "pcm_s16le", "sample_rate": 16000},
        "pcm_22050": {"container": "raw", "encoding": "pcm_s16le", "sample_rate": 22050},
        "pcm_44100": {"container": "raw", "encoding": "pcm_s16le", "sample_rate": 44100},
        "mulaw_8000": {"container": "raw", "encoding": "pcm_mulaw", "sample_rate": 8000},
        "alaw_8000": {"container": "raw", "encoding": "pcm_alaw", "sample_rate": 8000},
    }

    @classmethod
    def get_format(cls, format_name):
        if format_name in cls._format_mapping:
            return cls._format_mapping[format_name]
        else:
            raise ValueError(f"Unsupported format: {format_name}")
        
class VoiceMetadata(TypedDict):
    id: str
    name: str
    description: str
    embedding: List[float]
    is_public: bool
    user_id: str
    created_at: str
    
class OutputFormat(TypedDict):
    container: str
    encoding: str
    sample_rate: int
