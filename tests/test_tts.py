"""Test against the production Cartesia TTS API.

This test suite tries to be as general as possible because different keys will lead to
different results. Therefore, we cannot test for complete correctness but rather for
general correctness.
"""

import logging
import os
import sys
from cartesia import AsyncCartesia, Cartesia
from cartesia.client import DEFAULT_MODEL_ID, MULTILINGUAL_MODEL_ID
from cartesia._types import VoiceMetadata
from typing import AsyncGenerator, Generator, List
import numpy as np
import pytest

THISDIR = os.path.dirname(__file__)
sys.path.insert(0, os.path.dirname(THISDIR))

SAMPLE_VOICE = "Newsman"
SAMPLE_VOICE_ID = "d46abd1d-2d02-43e8-819f-51fb652c1c61"

logger = logging.getLogger(__name__)


class _Resources:
    def __init__(self, *, client: Cartesia, voices: List[VoiceMetadata], voice: VoiceMetadata):
        self.client = client
        self.voices = voices
        self.voice = voice


def create_client():
    return Cartesia(api_key=os.environ.get("CARTESIA_API_KEY"))


def create_async_client():
    return AsyncCartesia(api_key=os.environ.get("CARTESIA_API_KEY"))


@pytest.fixture(scope="session")
def client():
    logger.info("Creating client")
    return create_client()


@pytest.fixture(scope="session")
def resources(client: Cartesia):
    logger.info("Creating resources")
    voice = client.voices.get(SAMPLE_VOICE_ID)
    voices = client.voices.list()

    return _Resources(
        client=client,
        voices=voices,
        voice=voice
    )

def test_get_voices(client: Cartesia):
    logger.info("Testing voices.list")
    voices = client.voices.list()
    assert isinstance(voices, list)
    # Check that voices is a list of VoiceMetadata objects 
    assert all(isinstance(voice, dict) for voice in voices)
    ids = [voice["id"] for voice in voices]
    assert len(ids) == len(set(ids)), "All ids must be unique"

def test_get_voice_from_id(client: Cartesia):
    logger.info("Testing voices.get")
    voice = client.voices.get(SAMPLE_VOICE_ID)
    assert voice["id"] == SAMPLE_VOICE_ID
    assert voice["name"] == SAMPLE_VOICE
    assert voice["is_public"] is True
    voices = client.voices.list()
    assert voice in voices

# Does not work currently, LB issue
# def test_clone_voice_with_link(client: Cartesia):
#     url = "https://youtu.be/g2Z7Ddd573M?si=P8BM_hBqt5P8Ft6I&t=69"
#     logger.info("Testing voices.clone with link")
#     cloned_voice_embedding = client.voices.clone(link=url)
#     assert isinstance(cloned_voice_embedding, list)
#     assert len(cloned_voice_embedding) == 192
    
def test_create_voice(client: Cartesia):
    logger.info("Testing voices.create")
    embedding = np.ones(192).tolist()
    voice = client.voices.create(name="Test Voice", description="Test voice description", embedding=embedding)
    assert voice["name"] == "Test Voice"
    assert voice["description"] == "Test voice description"
    assert voice["is_public"] is False
    voices = client.voices.list()
    assert voice in voices

@pytest.mark.parametrize("stream", [True, False])
def test_sse_send(resources: _Resources, stream: bool):
    logger.info("Testing SSE send")
    client = resources.client
    transcript = "Hello, world! I'\''m generating audio on Cartesia."

    output_generate = client.tts.sse(transcript=transcript, voice_id=SAMPLE_VOICE_ID, output_format={
        "container": "raw",
        "encoding": "pcm_f32le",
        "sample_rate": 44100
    }, stream=stream, model_id=DEFAULT_MODEL_ID)
    
    if not stream:
        output_generate = [output_generate]

    for out in output_generate:
        assert isinstance(out["audio"], bytes)

@pytest.mark.parametrize("stream", [True, False])
def test_sse_send_with_model_id(resources: _Resources, stream: bool):
    logger.info("Testing SSE send with model_id")
    client = resources.client
    transcript = "Hello, world! I'\''m generating audio on Cartesia."

    output_generate = client.tts.sse(transcript=transcript, voice_id=SAMPLE_VOICE_ID, output_format={
        "container": "raw",
        "encoding": "pcm_f32le",
        "sample_rate": 44100
    }, stream=stream, model_id="upbeat-moon")
    
    if not stream:
        output_generate = [output_generate]

    for out in output_generate:
        assert isinstance(out["audio"], bytes)
        
@pytest.mark.parametrize("stream", [True, False])
def test_websocket_send(resources: _Resources, stream: bool):
    logger.info("Testing WebSocket send")
    client = resources.client
    transcript = "Hello, world! I'\''m generating audio on Cartesia."

    ws = client.tts.websocket()
    output_generate = ws.send(transcript=transcript, voice_id=SAMPLE_VOICE_ID, output_format={
        "container": "raw",
        "encoding": "pcm_f32le",
        "sample_rate": 44100
    }, stream=stream, model_id=DEFAULT_MODEL_ID)
    
    if not stream:
        output_generate = [output_generate]

    for out in output_generate:
        assert isinstance(out["audio"], bytes)
    
    ws.close()
        
def test_sse_send_context_manager(resources: _Resources):
    logger.info("Testing SSE send context manager")
    transcript = "Hello, world! I'\''m generating audio on Cartesia."

    with create_client() as client:
        output_generate = client.tts.sse(transcript=transcript, voice_id=SAMPLE_VOICE_ID, output_format={
            "container": "raw",
            "encoding": "pcm_f32le",
            "sample_rate": 44100
        }, stream=True, model_id=DEFAULT_MODEL_ID)
        assert isinstance(output_generate, Generator)
        
        for out in output_generate:
            assert out.keys() == {"audio"}
            assert isinstance(out["audio"], bytes)
            
def test_sse_send_context_manager_with_err():
    logger.info("Testing SSE send context manager with error")
    transcript = "Hello, world! I'\''m generating audio on Cartesia."
    
    try:
        with create_client() as client:
            client.tts.sse(transcript=transcript, voice_id="", output_format={
            "container": "raw",
            "encoding": "pcm_f32le",
            "sample_rate": 44100
            }, stream=True, model_id=DEFAULT_MODEL_ID) # should throw err because voice_id is ""
        raise RuntimeError("Expected error to be thrown")
    except Exception:
        pass
    
def test_websocket_send_context_manager(resources: _Resources):
    logger.info("Testing WebSocket send context manager")
    transcript = "Hello, world! I'\''m generating audio on Cartesia."

    with create_client() as client:
        ws = client.tts.websocket()
        output_generate = ws.send(transcript=transcript, voice_id=SAMPLE_VOICE_ID, output_format={
            "container": "raw",
            "encoding": "pcm_f32le",
            "sample_rate": 44100
        }, stream=True, model_id=DEFAULT_MODEL_ID)
        assert isinstance(output_generate, Generator)
        
        for out in output_generate:
            assert out.keys() == {"audio", "context_id"}
            assert isinstance(out["audio"], bytes)

def test_websocket_send_context_manage_err(resources: _Resources):
    logger.info("Testing WebSocket send context manager")
    transcript = "Hello, world! I'\''m generating audio on Cartesia."
    
    try:
        with create_client() as client:
            ws = client.tts.websocket()
            ws.send(transcript=transcript, voice_id="", output_format={
                "container": "raw",
                "encoding": "pcm_f32le",
                "sample_rate": 44100
            }, stream=True, model_id=DEFAULT_MODEL_ID) # should throw err because voice_id is ""
        raise RuntimeError("Expected error to be thrown")
    except Exception:
        pass
    
@pytest.mark.asyncio
async def test_async_sse_send(resources: _Resources):
    logger.info("Testing async SSE send")
    transcript = "Hello, world! I'\''m generating audio on Cartesia."

    async_client = create_async_client()
    try:
        output = await async_client.tts.sse(transcript=transcript, voice_id=SAMPLE_VOICE_ID, output_format={
            "container": "raw",
            "encoding": "pcm_f32le",
            "sample_rate": 44100
        }, stream=True, model_id=DEFAULT_MODEL_ID)
            
        async for out in output:
            assert out.keys() == {"audio"}
            assert isinstance(out["audio"], bytes)
    finally:
        # Close the websocket
        await async_client.close()
        
@pytest.mark.asyncio
async def test_async_websocket_send(resources: _Resources):
    logger.info("Testing async WebSocket send")
    transcript = "Hello, world! I'\''m generating audio on Cartesia."

    async_client = create_async_client()
    ws = await async_client.tts.websocket()
    try:
        output = await ws.send(transcript=transcript, voice_id=SAMPLE_VOICE_ID, output_format={
            "container": "raw",
            "encoding": "pcm_f32le",
            "sample_rate": 44100
        }, stream=True, model_id=DEFAULT_MODEL_ID)
        
        async for out in output:
            assert out.keys() == {"audio", "context_id"}
            assert isinstance(out["audio"], bytes)
    finally:
        # Close the websocket
        await ws.close()
        await async_client.close()
        
@pytest.mark.asyncio
async def test_async_sse_send_context_manager(resources: _Resources):
    logger.info("Testing async SSE send context manager")
    transcript = "Hello, world! I'\''m generating audio on Cartesia."

    async with create_async_client() as async_client:
        output_generate = await async_client.tts.sse(transcript=transcript, voice_id=SAMPLE_VOICE_ID, output_format={
            "container": "raw",
            "encoding": "pcm_f32le",
            "sample_rate": 44100
        }, stream=True, model_id=DEFAULT_MODEL_ID)
        assert isinstance(output_generate, AsyncGenerator)
        
        async for out in output_generate:
            assert out.keys() == {"audio"}
            assert isinstance(out["audio"], bytes)
   
@pytest.mark.asyncio         
async def test_async_sse_send_context_manager_with_err():
    logger.info("Testing async SSE send context manager with error")
    transcript = "Hello, world! I'\''m generating audio on Cartesia."
    
    try:
        async with create_async_client() as async_client:
            await async_client.tts.sse(transcript=transcript, voice_id="", output_format={
                "container": "raw",
                "encoding": "pcm_f32le",
                "sample_rate": 44100
            }, stream=True, model_id=DEFAULT_MODEL_ID) # should throw err because voice_id is ""
        raise RuntimeError("Expected error to be thrown")
    except Exception:
        pass
    
@pytest.mark.asyncio
async def test_async_websocket_send_context_manager():
    logger.info("Testing async WebSocket send context manager")
    transcript = "Hello, world! I'\''m generating audio on Cartesia."
    
    async with create_async_client() as async_client:
        ws = await async_client.tts.websocket()
        output_generate = await ws.send(transcript=transcript, voice_id=SAMPLE_VOICE_ID, output_format={
            "container": "raw",
            "encoding": "pcm_f32le",
            "sample_rate": 44100
        }, stream=True, model_id=DEFAULT_MODEL_ID)
        assert isinstance(output_generate, AsyncGenerator)
        
        async for out in output_generate:
            assert out.keys() == {"audio", "context_id"}
            assert isinstance(out["audio"], bytes)

@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize("language", ["en", "es", "fr", "de", "ja", "pt", "zh"])
def test_sse_send_multilingual(resources: _Resources, stream: bool, language: str):
    logger.info("Testing SSE send")
    client = resources.client
    transcript = "Hello, world! I'\''m generating audio on Cartesia."

    output_generate = client.tts.sse(transcript=transcript, voice_id=SAMPLE_VOICE_ID, output_format={
        "container": "raw",
        "encoding": "pcm_f32le",
        "sample_rate": 44100
    }, stream=stream, model_id=MULTILINGUAL_MODEL_ID, language=language)
    
    if not stream:
        output_generate = [output_generate]

    for out in output_generate:
        assert isinstance(out["audio"], bytes)
        
@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize("language", ["en", "es", "fr", "de", "ja", "pt", "zh"])
def test_websocket_send_multilingual(resources: _Resources, stream: bool, language: str):
    logger.info("Testing WebSocket send")
    client = resources.client
    transcript = "Hello, world! I'\''m generating audio on Cartesia."

    ws = client.tts.websocket()
    output_generate = ws.send(transcript=transcript, voice_id=SAMPLE_VOICE_ID, output_format={
        "container": "raw",
        "encoding": "pcm_f32le",
        "sample_rate": 44100
    }, stream=stream, model_id=MULTILINGUAL_MODEL_ID, language=language)
    
    if not stream:
        output_generate = [output_generate]

    for out in output_generate:
        assert isinstance(out["audio"], bytes)
    
    ws.close()
