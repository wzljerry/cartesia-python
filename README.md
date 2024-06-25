# Cartesia Python API Library

![PyPI - Version](https://img.shields.io/pypi/v/cartesia)
[![Discord](https://badgen.net/badge/black/Cartesia/icon?icon=discord&label)](https://discord.gg/ZVxavqHB9X)

The official Cartesia Python library which provides convenient access to the Cartesia REST and Websocket API from any Python 3.8+ application.

> [!IMPORTANT]
> The client library introduces breaking changes in v1.0.0, which was released on June 24th 2024. See the [release notes](https://github.com/cartesia-ai/cartesia-python/releases/tag/v1.0.0) and [migration guide](https://github.com/cartesia-ai/cartesia-python/discussions/44). Reach out to us on [Discord](https://discord.gg/ZVxavqHB9X) for any support requests!

## Documentation

Our complete API documentation can be found [on docs.cartesia.ai](https://docs.cartesia.ai).

## Installation

```bash
pip install cartesia

# pip install in editable mode w/ dev dependencies
pip install -e '.[dev]'
```

## Voices

```python
from cartesia import Cartesia
import os

client = Cartesia(api_key=os.environ.get("CARTESIA_API_KEY"))

# Get all available voices
voices = client.voices.list()
print(voices)

# Get a specific voice
voice = client.voices.get(id="a0e99841-438c-4a64-b679-ae501e7d6091")
print("The embedding for", voice["name"], "is", voice["embedding"])

# Clone a voice using filepath
cloned_voice_embedding = client.voices.clone(filepath="path/to/voice")

# Create a new voice
new_voice = client.voices.create(
    name="New Voice",
    description="A clone of my own voice",
    embedding=cloned_voice_embedding,
)
```

## Text-to-Speech

### Server-Sent Events (SSE)

```python
from cartesia import Cartesia
import pyaudio
import os

client = Cartesia(api_key=os.environ.get("CARTESIA_API_KEY"))
voice_name = "Barbershop Man"
voice_id = "a0e99841-438c-4a64-b679-ae501e7d6091"
voice = client.voices.get(id=voice_id)

transcript = "Hello! Welcome to Cartesia"

# You can check out our models at [docs.cartesia.ai](https://docs.cartesia.ai/getting-started/available-models).
model_id = "sonic-english"

# You can find the supported `output_format`s in our [API Reference](https://docs.cartesia.ai/api-reference/endpoints/stream-speech-server-sent-events).
output_format = {
    "container": "raw",
    "encoding": "pcm_f32le",
    "sample_rate": 44100,
}

p = pyaudio.PyAudio()
rate = 44100

stream = None

# Generate and stream audio
for output in client.tts.sse(
    model_id=model_id,
    transcript=transcript,
    voice_embedding=voice["embedding"],
    stream=True,
    output_format=output_format,
):
    buffer = output["audio"]

    if not stream:
        stream = p.open(format=pyaudio.paFloat32, channels=1, rate=rate, output=True)

    # Write the audio data to the stream
    stream.write(buffer)

stream.stop_stream()
stream.close()
p.terminate()
```

You can also use the async client if you want to make asynchronous API calls. Simply import `AsyncCartesia` instead of `Cartesia` and use await with each API call:

```python
from cartesia import AsyncCartesia
import asyncio
import pyaudio
import os


async def write_stream():
    client = AsyncCartesia(api_key=os.environ.get("CARTESIA_API_KEY"))
    voice_name = "Barbershop Man"
    voice_id = "a0e99841-438c-4a64-b679-ae501e7d6091"
    voice = client.voices.get(id=voice_id)
    transcript = "Hello! Welcome to Cartesia"
    # You can check out our models at [docs.cartesia.ai](https://docs.cartesia.ai/getting-started/available-models).
    model_id = "sonic-english"

    # You can find the supported `output_format`s in our [API Reference](https://docs.cartesia.ai/api-reference/endpoints/stream-speech-server-sent-events).
    output_format = {
        "container": "raw",
        "encoding": "pcm_f32le",
        "sample_rate": 44100,
    }

    p = pyaudio.PyAudio()
    rate = 44100

    stream = None

    # Generate and stream audio
    async for output in await client.tts.sse(
        model_id=model_id,
        transcript=transcript,
        voice_embedding=voice["embedding"],
        stream=True,
        output_format=output_format,
    ):
        buffer = output["audio"]

        if not stream:
            stream = p.open(
                format=pyaudio.paFloat32, channels=1, rate=rate, output=True
            )

        # Write the audio data to the stream
        stream.write(buffer)

    stream.stop_stream()
    stream.close()
    p.terminate()
    await client.close()


asyncio.run(write_stream())
```

### WebSocket

```python
from cartesia import Cartesia
import pyaudio
import os

client = Cartesia(api_key=os.environ.get("CARTESIA_API_KEY"))
voice_name = "Barbershop Man"
voice_id = "a0e99841-438c-4a64-b679-ae501e7d6091"
voice = client.voices.get(id=voice_id)
transcript = "Hello! Welcome to Cartesia"

# You can check out our models at [docs.cartesia.ai](https://docs.cartesia.ai/getting-started/available-models).
model_id = "sonic-english"

# You can find the supported `output_format`s in our [API Reference](https://docs.cartesia.ai/api-reference/endpoints/stream-speech-server-sent-events).
output_format = {
    "container": "raw",
    "encoding": "pcm_f32le",
    "sample_rate": 22050,
}

p = pyaudio.PyAudio()
rate = 22050

stream = None

# Set up the websocket connection
ws = client.tts.websocket()

# Generate and stream audio using the websocket
for output in ws.send(
    model_id=model_id,
    transcript=transcript,
    voice_embedding=voice["embedding"],
    stream=True,
    output_format=output_format,
):
    buffer = output["audio"]

    if not stream:
        stream = p.open(format=pyaudio.paFloat32, channels=1, rate=rate, output=True)

    # Write the audio data to the stream
    stream.write(buffer)

stream.stop_stream()
stream.close()
p.terminate()

ws.close()  # Close the websocket connection
```

### Multilingual Text-to-Speech [Alpha]

You can use our `sonic-multilingual` model to generate audio in multiple languages. The languages supported are available at [docs.cartesia.ai](https://docs.cartesia.ai/getting-started/available-models).

```python
from cartesia import Cartesia
import pyaudio
import os

client = Cartesia(api_key=os.environ.get("CARTESIA_API_KEY"))
voice_name = "Barbershop Man"
voice_id = "a0e99841-438c-4a64-b679-ae501e7d6091"
voice = client.voices.get(id=voice_id)

transcript = "Hola! Bienvenido a Cartesia"
language = "es"  # Language code corresponding to the language of the transcript

# Make sure you use the multilingual model! You can check out all models at [docs.cartesia.ai](https://docs.cartesia.ai/getting-started/available-models).
model_id = "sonic-multilingual"

# You can find the supported `output_format`s in our [API Reference](https://docs.cartesia.ai/api-reference/endpoints/stream-speech-server-sent-events).
output_format = {
    "container": "raw",
    "encoding": "pcm_f32le",
    "sample_rate": 44100,
}

p = pyaudio.PyAudio()
rate = 44100

stream = None

# Pass in the corresponding language code to the `language` parameter to generate and stream audio.
for output in client.tts.sse(
    model_id=model_id,
    transcript=transcript,
    voice_embedding=voice["embedding"],
    stream=True,
    output_format=output_format,
    language=language,
):
    buffer = output["audio"]

    if not stream:
        stream = p.open(format=pyaudio.paFloat32, channels=1, rate=rate, output=True)

    stream.write(buffer)

stream.stop_stream()
stream.close()
p.terminate()
```

If you are using Jupyter Notebook or JupyterLab, you can use IPython.display.Audio to play the generated audio directly in the notebook.
Additionally, in these notebook examples we show how to use the client as a context manager (though this is not required).

```python
from IPython.display import Audio
import io
import os
import numpy as np

from cartesia import Cartesia

with Cartesia(api_key=os.environ.get("CARTESIA_API_KEY")) as client:
    output_format = {
        "container": "raw",
        "encoding": "pcm_f32le",
        "sample_rate": 8000,
    }
    rate = 8000
    voice_id = "a0e99841-438c-4a64-b679-ae501e7d6091"
    voice = client.voices.get(id=voice_id)
    transcript = "Hey there! Welcome to Cartesia"

    # Create a BytesIO object to store the audio data
    audio_data = io.BytesIO()

    # Generate and stream audio
    for output in client.tts.sse(
        model_id="sonic-english",
        transcript=transcript,
        voice_embedding=voice["embedding"],
        stream=True,
        output_format=output_format,
    ):
        buffer = output["audio"]
        audio_data.write(buffer)

# Set the cursor position to the beginning of the BytesIO object
audio_data.seek(0)

# Create an Audio object from the BytesIO data
audio = Audio(np.frombuffer(audio_data.read(), dtype=np.float32), rate=rate)

# Display the Audio object
display(audio)
```

Below is the same example using the async client:

```python
from IPython.display import Audio
import io
import os
import numpy as np

from cartesia import AsyncCartesia

async with AsyncCartesia(api_key=os.environ.get("CARTESIA_API_KEY")) as client:
    output_format = {
        "container": "raw",
        "encoding": "pcm_f32le",
        "sample_rate": 8000,
    }
    rate = 8000
    voice_id = "248be419-c632-4f23-adf1-5324ed7dbf1d"
    transcript = "Hey there! Welcome to Cartesia"

    # Create a BytesIO object to store the audio data
    audio_data = io.BytesIO()

    # Generate and stream audio
    async for output in client.tts.sse(
        model_id="sonic-english",
        transcript=transcript,
        voice_id=voice_id,
        stream=True,
        output_format=output_format,
    ):
        buffer = output["audio"]
        audio_data.write(buffer)

# Set the cursor position to the beginning of the BytesIO object
audio_data.seek(0)

# Create an Audio object from the BytesIO data
audio = Audio(np.frombuffer(audio_data.read(), dtype=np.float32), rate=rate)

# Display the Audio object
display(audio)
```

### Utility methods

#### Output Formats

You can use the `client.tts.get_output_format` method to convert string-based output format names into the `output_format` dictionary which is expected by the `output_format` parameter. You can see the `OutputFormatMapping` class in `cartesia._types` for the currently supported output format names. You can also view the currently supported `output_format`s in our [API Reference](https://docs.cartesia.ai/api-reference/endpoints/stream-speech-server-sent-events).

The previously used `output_format` strings are now deprecated and will be removed in v1.2.0. These are listed in the `DeprecatedOutputFormatMapping` class in `cartesia._types`.

```python
# Get the output format dictionary from string name
output_format = client.tts.get_output_format("raw_pcm_f32le_44100")

# Pass in the output format dictionary to generate and stream audio
generator = client.tts.sse(
    model_id=model,
    transcript=transcript,
    voice_id=SAMPLE_VOICE_ID,
    stream=True,
    output_format=output_format,
)
```

To avoid storing your API key in the source code, we recommend doing one of the following:

1. Use [`python-dotenv`](https://pypi.org/project/python-dotenv/) to add `CARTESIA_API_KEY="my-api-key"` to your .env file.
1. Set the `CARTESIA_API_KEY` environment variable, preferably to a secure shell init file (e.g. `~/.zshrc`, `~/.bashrc`)
