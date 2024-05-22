"""Test against the production Cartesia TTS API.

This test suite tries to be as general as possible because different keys will lead to
different results. Therefore, we cannot test for complete correctness but rather for
general correctness.
"""

import logging
import os
import sys
from cartesia.tts import DEFAULT_MODEL_ID, AsyncCartesiaTTS, CartesiaTTS
from cartesia._types import AudioDataReturnType, AudioOutputFormat, VoiceMetadata
from typing import AsyncGenerator, Dict, Generator, List, Optional, Union
import numpy as np
import pytest

THISDIR = os.path.dirname(__file__)
sys.path.insert(0, os.path.dirname(THISDIR))

SAMPLE_VOICE = "Abby"

logger = logging.getLogger(__name__)


class _Resources:
    def __init__(self, *, client: CartesiaTTS, voices: Dict[str, VoiceMetadata]):
        self.client = client
        self.voices = voices


def create_client():
    return CartesiaTTS(api_key=os.environ.get("CARTESIA_API_KEY"))


def create_async_client():
    return AsyncCartesiaTTS(api_key=os.environ.get("CARTESIA_API_KEY"))


@pytest.fixture(scope="session")
def client():
    logger.info("Creating client")
    return create_client()


@pytest.fixture(scope="session")
def resources(client: CartesiaTTS):
    logger.info("Creating resources")
    voices = client.get_voices()
    voice_id = voices[SAMPLE_VOICE]["id"]
    voices[SAMPLE_VOICE]["embedding"] = client.get_voice_embedding(voice_id=voice_id)

    return _Resources(
        client=client,
        voices=voices,
    )


def test_get_voices(client: CartesiaTTS):
    logger.info("Testing get_voices")
    voices = client.get_voices()

    assert isinstance(voices, dict)
    assert all(isinstance(key, str) for key in voices.keys())
    ids = [voice["id"] for voice in voices.values()]
    assert len(ids) == len(set(ids)), "All ids must be unique"
    assert all(
        key == voice["name"] for key, voice in voices.items()
    ), "The key must be the same as the name"


def test_get_voice_embedding_from_id(client: CartesiaTTS):
    logger.info("Testing get_voice_embedding")
    voices = client.get_voices()
    voice_id = voices[SAMPLE_VOICE]["id"]

    client.get_voice_embedding(voice_id=voice_id)


def test_get_voice_embedding_from_url(client: CartesiaTTS):
    url = "https://youtu.be/g2Z7Ddd573M?si=P8BM_hBqt5P8Ft6I&t=69"
    logger.info(f"Testing get_voice_embedding from URL {url}")
    client.get_voice_embedding(link=url)


@pytest.mark.parametrize("websocket", [True, False])
def test_generate(resources: _Resources, websocket: bool):
    logger.info("Testing generate")
    client = resources.client
    voices = resources.voices
    embedding = voices[SAMPLE_VOICE]["embedding"]
    transcript = "Hello, world!"

    output = client.generate(transcript=transcript, voice=embedding, websocket=websocket)
    assert output.keys() == {"audio", "sampling_rate"}
    assert isinstance(output["audio"], bytes)
    assert isinstance(output["sampling_rate"], int)


def test_generate_with_model_id(resources: _Resources):
    logger.info("Testing generate with model_id")
    client = resources.client
    voices = resources.voices
    embedding = voices[SAMPLE_VOICE]["embedding"]
    transcript = "Hello, world!"

    output = client.generate(transcript=transcript, voice=embedding, model_id=DEFAULT_MODEL_ID)
    assert output.keys() == {"audio", "sampling_rate"}
    assert isinstance(output["audio"], bytes)
    assert isinstance(output["sampling_rate"], int)


@pytest.mark.parametrize("websocket", [True, False])
def test_generate_stream(resources: _Resources, websocket: bool):
    logger.info("Testing generate stream")
    client = resources.client
    voices = resources.voices
    embedding = voices[SAMPLE_VOICE]["embedding"]
    transcript = "Hello, world!"

    generator = client.generate(
        transcript=transcript, voice=embedding, websocket=websocket, stream=True
    )
    assert isinstance(generator, Generator)

    for output in generator:
        assert output.keys() == {"audio", "sampling_rate"}
        assert isinstance(output["audio"], bytes)
        assert isinstance(output["sampling_rate"], int)


@pytest.mark.parametrize("websocket", [True, False])
def test_generate_stream_context_manager(resources: _Resources, websocket: bool):
    logger.info("Testing generate stream context manager")
    voices = resources.voices
    embedding = voices[SAMPLE_VOICE]["embedding"]
    transcript = "Hello, world!"

    with create_client() as client:
        generator = client.generate(
            transcript=transcript, voice=embedding, websocket=websocket, stream=True
        )
        assert isinstance(generator, Generator)

        for output in generator:
            assert output.keys() == {"audio", "sampling_rate"}
            assert isinstance(output["audio"], bytes)
            assert isinstance(output["sampling_rate"], int)


def test_generate_context_manager_with_err():
    logger.info("Testing generate context manager with error")
    websocket = None
    websocket_was_opened = False
    try:
        with create_client() as client:
            client.refresh_websocket()
            websocket = client.websocket
            websocket_was_opened = websocket.socket.fileno() != -1
            client.generate(
                transcript=None, voice=None, websocket=True
            )  # should throw because transcript None
        raise RuntimeError("Expected AttributeError to be thrown")
    except AttributeError:
        pass

    assert websocket_was_opened
    assert websocket.socket.fileno() == -1  # check socket is now closed


@pytest.mark.parametrize("output_format", [_fmt for _fmt in AudioOutputFormat])
@pytest.mark.parametrize("as_str", [True, False])
@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize("websocket", [True, False])
def test_generate_with_output_format(
    resources: _Resources,
    output_format: AudioOutputFormat,
    as_str: bool,
    stream: bool,
    websocket: bool,
):
    value = output_format.value

    client = resources.client
    voices = resources.voices
    embedding = voices[SAMPLE_VOICE]["embedding"]
    transcript = "Hello, world!"

    split = value.split("_")
    expected_sampling_rate = int(split[1]) if len(split) == 2 else 44_100

    # Easy way to get around iterating over stream=True / False.
    output_generate = client.generate(
        transcript=transcript,
        voice=embedding,
        websocket=websocket,
        stream=stream,
        output_format=output_format.value if as_str else output_format,
    )
    if not stream:
        output_generate = [output_generate]

    for out in output_generate:
        assert isinstance(out["audio"], bytes)
        assert out["sampling_rate"] == expected_sampling_rate


@pytest.mark.parametrize("data_rtype", [_fmt for _fmt in AudioDataReturnType])
@pytest.mark.parametrize("as_str", [True, False])
@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize("websocket", [True, False])
def test_generate_with_data_rtype(
    resources: _Resources,
    data_rtype: AudioDataReturnType,
    as_str: bool,
    stream: bool,
    websocket: bool,
):
    client = resources.client
    voices = resources.voices
    embedding = voices[SAMPLE_VOICE]["embedding"]
    transcript = "Hello, world!"

    # Easy way to get around iterating over stream=True / False.
    output_generate = client.generate(
        transcript=transcript,
        voice=embedding,
        websocket=websocket,
        stream=stream,
        data_rtype=data_rtype.value if as_str else data_rtype,
    )
    if not stream:
        output_generate = [output_generate]

    for out in output_generate:
        if data_rtype == AudioDataReturnType.BYTES:
            assert isinstance(out["audio"], bytes)
        elif data_rtype == AudioDataReturnType.ARRAY:
            assert isinstance(out["audio"], np.ndarray)
            assert out["audio"].dtype == np.float32


@pytest.mark.parametrize("websocket", [True, False])
@pytest.mark.asyncio
async def test_async_generate(resources: _Resources, websocket: bool):
    logger.info("Testing async generate")
    voices = resources.voices
    embedding = voices[SAMPLE_VOICE]["embedding"]
    transcript = "Hello, world!"

    async_client = create_async_client()
    try:
        output = await async_client.generate(
            transcript=transcript, voice=embedding, websocket=websocket
        )

        assert output.keys() == {"audio", "sampling_rate"}
        assert isinstance(output["audio"], bytes)
        assert isinstance(output["sampling_rate"], int)
    finally:
        # Close the websocket
        await async_client.close()


@pytest.mark.parametrize("websocket", [True, False])
@pytest.mark.asyncio
async def test_async_generate_stream(resources: _Resources, websocket: bool):
    logger.info(f"Testing async generate stream with websocket={websocket}")
    voices = resources.voices
    embedding = voices[SAMPLE_VOICE]["embedding"]
    transcript = "Hello, world!"

    async_client = create_async_client()

    try:
        generator = await async_client.generate(
            transcript=transcript, voice=embedding, websocket=websocket, stream=True
        )
        assert isinstance(generator, AsyncGenerator)
        async for output in generator:
            assert output.keys() == {"audio", "sampling_rate"}
            assert isinstance(output["audio"], bytes)
            assert isinstance(output["sampling_rate"], int)
    finally:
        # Close the websocket
        await async_client.close()


@pytest.mark.parametrize("websocket", [True, False])
@pytest.mark.asyncio
async def test_async_generate_stream_context_manager(resources: _Resources, websocket: bool):
    logger.info("Testing async generate stream context manager")
    voices = resources.voices
    embedding = voices[SAMPLE_VOICE]["embedding"]
    transcript = "Hello, world!"

    async with create_async_client() as async_client:
        generator = await async_client.generate(transcript=transcript, voice=embedding, stream=True)
        assert isinstance(generator, AsyncGenerator)

        async for output in generator:
            assert output.keys() == {"audio", "sampling_rate"}
            assert isinstance(output["audio"], bytes)
            assert isinstance(output["sampling_rate"], int)


@pytest.mark.asyncio
async def test_generate_async_context_manager_with_err():
    logger.info("Testing async generate context manager with error")
    websocket = None
    websocket_was_opened = False
    try:
        async with create_async_client() as async_client:
            await async_client.refresh_websocket()
            websocket = async_client.websocket
            websocket_was_opened = not websocket.closed
            # below should throw because transcript None
            await async_client.generate(transcript=None, voice=None, websocket=True)
        raise RuntimeError("Expected AttributeError to be thrown")
    except AttributeError:
        pass

    assert websocket_was_opened
    assert websocket.closed  # check websocket is now closed


@pytest.mark.parametrize("output_format", [_fmt for _fmt in AudioOutputFormat])
@pytest.mark.parametrize("as_str", [True, False])
@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.asyncio
async def test_generate_async_with_output_format(
    resources: _Resources, output_format: AudioOutputFormat, as_str: bool, stream: bool
):
    logger.info(
        f"Testing async generate stream with output_format={output_format}, as_str={as_str}, stream={stream}"
    )
    voices = resources.voices
    embedding = voices[SAMPLE_VOICE]["embedding"]
    transcript = "Hello, world!"

    split = output_format.value.split("_")
    expected_sampling_rate = int(split[1]) if len(split) == 2 else 44_100

    def _validate(output):
        assert isinstance(output["audio"], bytes)
        assert output["sampling_rate"] == expected_sampling_rate

    async_client = create_async_client()

    try:
        output_generate = await async_client.generate(
            transcript=transcript,
            voice=embedding,
            websocket=False,
            stream=stream,
            output_format=output_format.value if as_str else output_format,
        )
        if stream:
            generator = output_generate
            assert isinstance(generator, AsyncGenerator)
            async for output in generator:
                _validate(output)
        else:
            _validate(output_generate)
    finally:
        # Close the websocket
        await async_client.close()


@pytest.mark.parametrize("data_rtype", [_fmt for _fmt in AudioDataReturnType])
@pytest.mark.parametrize("as_str", [True, False])
@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize("websocket", [True, False])
@pytest.mark.asyncio
async def test_generate_async_with_data_rtype(resources: _Resources, data_rtype: AudioDataReturnType, as_str: bool, stream: bool, websocket: bool):
    voices = resources.voices
    embedding = voices[SAMPLE_VOICE]["embedding"]
    transcript = "Hello, world!"

    async_client = create_async_client()

    def _validate(output):
        if data_rtype == AudioDataReturnType.BYTES:
            assert isinstance(output["audio"], bytes)
        elif data_rtype == AudioDataReturnType.ARRAY:
            assert isinstance(output["audio"], np.ndarray)
            assert output["audio"].dtype == np.float32

    try:
        output_generate = await async_client.generate(
            transcript=transcript,
            voice=embedding,
            websocket=False,
            stream=stream,
            data_rtype=data_rtype.value if as_str else data_rtype,
        )
        if stream:
            assert isinstance(output_generate, AsyncGenerator)
            async for output in output_generate:
                _validate(output)
        else:
            _validate(output_generate)
    finally:
        # Close the websocket
        await async_client.close()


@pytest.mark.parametrize("chunk_time", [0.05, 0.6])
def test_check_inputs_invalid_chunk_time(client: CartesiaTTS, chunk_time):
    logger.info(f"Testing invalid chunk_time: {chunk_time}")
    with pytest.raises(ValueError, match="`chunk_time` must be between 0.1 and 0.5"):
        client._check_inputs(
            "Test",
            None,
            chunk_time,
            output_format=AudioOutputFormat.FP32,
            data_rtype=AudioDataReturnType.BYTES,
        )


@pytest.mark.parametrize("chunk_time", [0.1, 0.3, 0.5])
def test_check_inputs_valid_chunk_time(client, chunk_time):
    logger.info("Testing valid chunk_time: {chunk_time}")
    try:
        client._check_inputs(
            "Test",
            None,
            chunk_time,
            output_format=AudioOutputFormat.FP32,
            data_rtype=AudioDataReturnType.BYTES,
        )
    except ValueError:
        pytest.fail("Unexpected ValueError raised")


def test_check_inputs_duration_less_than_chunk_time(client: CartesiaTTS):
    logger.info("Testing duration less than chunk_time")
    with pytest.raises(ValueError, match="`duration` must be greater than chunk_time"):
        client._check_inputs(
            "Test",
            0.2,
            0.3,
            output_format=AudioOutputFormat.FP32,
            data_rtype=AudioDataReturnType.BYTES,
        )


@pytest.mark.parametrize("duration,chunk_time", [(0.5, 0.2), (1.0, 0.5), (2.0, 0.1)])
def test_check_inputs_valid_duration_and_chunk_time(client: CartesiaTTS, duration, chunk_time):
    logger.info(f"Testing valid duration: {duration} and chunk_time: {chunk_time}")
    try:
        client._check_inputs(
            "Test",
            duration,
            chunk_time,
            output_format=AudioOutputFormat.FP32,
            data_rtype=AudioDataReturnType.BYTES,
        )
    except ValueError:
        pytest.fail("Unexpected ValueError raised")


def test_check_inputs_empty_transcript(client: CartesiaTTS):
    logger.info("Testing empty transcript")
    with pytest.raises(ValueError, match="`transcript` must be non empty"):
        client._check_inputs(
            "",
            None,
            None,
            output_format=AudioOutputFormat.FP32,
            data_rtype=AudioDataReturnType.BYTES,
        )


@pytest.mark.parametrize("transcript", ["Hello", "Test transcript", "Lorem ipsum dolor sit amet"])
def test_check_inputs_valid_transcript(client: CartesiaTTS, transcript):
    logger.info(f"Testing valid transcript: {transcript}")
    try:
        client._check_inputs(
            transcript,
            None,
            None,
            output_format=AudioOutputFormat.FP32,
            data_rtype=AudioDataReturnType.BYTES,
        )
    except ValueError:
        pytest.fail("Unexpected ValueError raised")


@pytest.mark.parametrize(
    "output_format,error",
    [
        # Valid output formats.
        ("fp32", None),
        ("pcm", None),
        ("fp32_16000", None),
        ("fp32_22050", None),
        ("fp32_44100", None),
        ("pcm_16000", None),
        ("pcm_22050", None),
        ("pcm_44100", None),
        # Invalid output formats.
        ("invalid", ValueError),
        ("pcm_1234", ValueError),  # cannot specify arbitrary sampling rate
        ("fp32_1234", ValueError),  # cannot specify arbitrary sampling rate
        ("fp16_44100", ValueError),  # fp16 not supported.
    ],
)
def test_check_inputs_output_format(
    client: CartesiaTTS, output_format: Union[str, AudioOutputFormat], error: Optional[Exception]
):
    if error:
        with pytest.raises(error):
            client._check_inputs(
                "Test",
                None,
                None,
                output_format=output_format,
                data_rtype=AudioDataReturnType.BYTES,
            )
    else:
        client._check_inputs(
            "Test",
            None,
            None,
            output_format=output_format,
            data_rtype=AudioDataReturnType.BYTES,
        )


@pytest.mark.parametrize(
    "data_rtype,error",
    [
        # Valid data return types.
        ("bytes", None),
        ("array", None),
        # Invalid data return types.
        ("invalid", ValueError),
        ("tensor", ValueError),
    ],
)
def test_check_inputs_data_rtype(
    client: CartesiaTTS, data_rtype: Union[str, AudioDataReturnType], error: Optional[Exception]
):
    if error:
        with pytest.raises(error):
            client._check_inputs(
                "Test",
                None,
                None,
                output_format=AudioOutputFormat.FP32,
                data_rtype=data_rtype,
            )
    else:
        client._check_inputs(
            "Test",
            None,
            None,
            output_format=AudioOutputFormat.FP32,
            data_rtype=data_rtype,
        )
