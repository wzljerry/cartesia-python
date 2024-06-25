import asyncio
import base64
import json
import os
import uuid
from types import TracebackType
from typing import Any, AsyncGenerator, Dict, Generator, List, Optional, Tuple, Union, Callable

import aiohttp
import httpx
import logging
import requests
from websockets.sync.client import connect

from cartesia.utils.retry import retry_on_connection_error, retry_on_connection_error_async
from cartesia.utils.deprecated import deprecated
from cartesia._types import (
    OutputFormat,
    OutputFormatMapping,
    DeprecatedOutputFormatMapping,
    VoiceMetadata,
)


DEFAULT_MODEL_ID = "sonic-english"  # latest default model
MULTILINGUAL_MODEL_ID = "sonic-multilingual"  # latest multilingual model
DEFAULT_BASE_URL = "api.cartesia.ai"
DEFAULT_CARTESIA_VERSION = "2024-06-10"  # latest version
DEFAULT_TIMEOUT = 30  # seconds
DEFAULT_NUM_CONNECTIONS = 10  # connections per client

BACKOFF_FACTOR = 1
MAX_RETRIES = 3

logger = logging.getLogger(__name__)


class BaseClient:
    def __init__(self, *, api_key: Optional[str] = None, timeout: float = DEFAULT_TIMEOUT):
        """Constructor for the BaseClient. Used by the Cartesia and AsyncCartesia clients."""
        self.api_key = api_key or os.environ.get("CARTESIA_API_KEY")
        self.timeout = timeout


class Resource:
    def __init__(
        self,
        api_key: str,
        timeout: float,
    ):
        """Constructor for the Resource class. Used by the Voices and TTS classes."""
        self.api_key = api_key
        self.timeout = timeout
        self.base_url = os.environ.get("CARTESIA_BASE_URL", DEFAULT_BASE_URL)
        self.cartesia_version = DEFAULT_CARTESIA_VERSION
        self.headers = {
            "X-API-Key": self.api_key,
            "Cartesia-Version": self.cartesia_version,
            "Content-Type": "application/json",
        }

    def _http_url(self):
        """Returns the HTTP URL for the Cartesia API.
        If the base URL is localhost, the URL will start with 'http'. Otherwise, it will start with 'https'.
        """
        if self.base_url.startswith("http://") or self.base_url.startswith("https://"):
            return self.base_url
        else:
            prefix = "http" if "localhost" in self.base_url else "https"
            return f"{prefix}://{self.base_url}"

    def _ws_url(self):
        """Returns the WebSocket URL for the Cartesia API.
        If the base URL is localhost, the URL will start with 'ws'. Otherwise, it will start with 'wss'.
        """
        if self.base_url.startswith("ws://") or self.base_url.startswith("wss://"):
            return self.base_url
        else:
            prefix = "ws" if "localhost" in self.base_url else "wss"
            return f"{prefix}://{self.base_url}"


class Cartesia(BaseClient):
    """
    The client for Cartesia's text-to-speech library.

    This client contains methods to interact with the Cartesia text-to-speech API.
    The client can be used to manage your voice library and generate speech from text.

    The client supports generating audio using both Server-Sent Events and WebSocket for lower latency.
    """

    def __init__(self, *, api_key: Optional[str] = None, timeout: float = DEFAULT_TIMEOUT):
        """Constructor for the Cartesia client.

        Args:
            api_key: The API key to use for authorization.
                If not specified, the API key will be read from the environment variable
                `CARTESIA_API_KEY`.
            timeout: The timeout for the HTTP requests in seconds. Defaults to 30 seconds.
        """
        super().__init__(api_key=api_key, timeout=timeout)
        self.voices = Voices(api_key=self.api_key, timeout=self.timeout)
        self.tts = TTS(api_key=self.api_key, timeout=self.timeout)

    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type: Union[type, None],
        exc: Union[BaseException, None],
        exc_tb: Union[TracebackType, None],
    ):
        pass


class Voices(Resource):
    """This resource contains methods to list, get, clone, and create voices in your Cartesia voice library.

    Usage:
        >>> client = Cartesia(api_key="your_api_key")
        >>> voices = client.voices.list()
        >>> voice = client.voices.get(id="a0e99841-438c-4a64-b679-ae501e7d6091")
        >>> print("Voice Name:", voice["name"], "Voice Description:", voice["description"])
        >>> embedding = client.voices.clone(filepath="path/to/clip.wav")
        >>> new_voice = client.voices.create(
        ...     name="My Voice", description="A new voice", embedding=embedding
        ... )
    """

    def list(self) -> List[VoiceMetadata]:
        """List all voices in your voice library.

        Returns:
        This method returns a list of VoiceMetadata objects.
        """
        response = httpx.get(
            f"{self._http_url()}/voices",
            headers=self.headers,
            timeout=self.timeout,
        )

        if not response.is_success:
            raise ValueError(f"Failed to get voices. Error: {response.text}")

        voices = response.json()
        return voices

    def get(self, id: str) -> VoiceMetadata:
        """Get a voice by its ID.

        Args:
            id: The ID of the voice.

        Returns:
            A VoiceMetadata object containing the voice metadata.
        """
        url = f"{self._http_url()}/voices/{id}"
        response = httpx.get(url, headers=self.headers, timeout=self.timeout)

        if not response.is_success:
            raise ValueError(
                f"Failed to get voice. Status Code: {response.status_code}\n"
                f"Error: {response.text}"
            )

        return response.json()

    def clone(self, filepath: Optional[str] = None, link: Optional[str] = None) -> List[float]:
        """Clone a voice from a clip or a URL.

        Args:
            filepath: The path to the clip file.
            link: The URL to the clip

        Returns:
            The embedding of the cloned voice as a list of floats.
        """
        # TODO: Python has a bytes object, use that instead of a filepath
        if not filepath and not link:
            raise ValueError("At least one of 'filepath' or 'link' must be specified.")
        if filepath and link:
            raise ValueError("Only one of 'filepath' or 'link' should be specified.")
        if filepath:
            url = f"{self._http_url()}/voices/clone/clip"
            with open(filepath, "rb") as file:
                files = {"clip": file}
                headers = self.headers.copy()
                headers.pop("Content-Type", None)
                headers["Content-Type"] = "multipart/form-data"
                response = httpx.post(url, headers=headers, files=files, timeout=self.timeout)
                if not response.is_success:
                    raise ValueError(f"Failed to clone voice from clip. Error: {response.text}")
        elif link:
            url = f"{self._http_url()}/voices/clone/url"
            params = {"link": link}
            headers = self.headers.copy()
            headers.pop("Content-Type")  # The content type header is not required for URLs
            response = httpx.post(url, headers=self.headers, params=params, timeout=self.timeout)
            if not response.is_success:
                raise ValueError(f"Failed to clone voice from URL. Error: {response.text}")

        return response.json()["embedding"]

    def create(self, name: str, description: str, embedding: List[float]) -> VoiceMetadata:
        """Create a new voice.

        Args:
            name: The name of the voice.
            description: The description of the voice.
            embedding: The embedding of the voice. This should be generated with :meth:`clone`.

        Returns:
            A dictionary containing the voice metadata.
        """
        response = httpx.post(
            f"{self._http_url()}/voices",
            headers=self.headers,
            json={"name": name, "description": description, "embedding": embedding},
            timeout=self.timeout,
        )

        if not response.is_success:
            raise ValueError(f"Failed to create voice. Error: {response.text}")

        return response.json()


class _WebSocket:
    """This class contains methods to generate audio using WebSocket. Ideal for low-latency audio generation.

    Usage:
        >>> ws = client.tts.websocket()
        >>> for audio_chunk in ws.send(
        ...     model_id="upbeat-moon", transcript="Hello world!", voice_embedding=embedding,
        ...     output_format={"container": "raw", "encoding": "pcm_f32le", "sample_rate": 44100}, stream=True
        ... ):
        ...     audio = audio_chunk["audio"]
    """

    def __init__(
        self,
        ws_url: str,
        api_key: str,
        cartesia_version: str,
    ):
        self.ws_url = ws_url
        self.api_key = api_key
        self.cartesia_version = cartesia_version
        self.websocket = None

    def connect(self):
        """This method connects to the WebSocket if it is not already connected."""
        if self.websocket is None or self._is_websocket_closed():
            route = "tts/websocket"
            self.websocket = connect(
                f"{self.ws_url}/{route}?api_key={self.api_key}&cartesia_version={self.cartesia_version}"
            )

    def _is_websocket_closed(self):
        return self.websocket.socket.fileno() == -1

    def close(self):
        """This method closes the WebSocket connection. *Highly* recommended to call this method when done using the WebSocket."""
        if self.websocket is not None and not self._is_websocket_closed():
            self.websocket.close()

    def _convert_response(
        self, response: Dict[str, any], include_context_id: bool
    ) -> Dict[str, Any]:
        audio = base64.b64decode(response["data"])

        optional_kwargs = {}
        if include_context_id:
            optional_kwargs["context_id"] = response["context_id"]

        return {
            "audio": audio,
            **optional_kwargs,
        }

    def _validate_and_construct_voice(
        self, voice_id: Optional[str] = None, voice_embedding: Optional[List[float]] = None
    ) -> dict:
        """Validate and construct the voice dictionary for the request.

        Args:
            voice_id: The ID of the voice to use for generating audio.
            voice_embedding: The embedding of the voice to use for generating audio.

        Returns:
            A dictionary representing the voice configuration.

        Raises:
            ValueError: If neither or both voice_id and voice_embedding are specified.
        """
        if voice_id is None and voice_embedding is None:
            raise ValueError("Either voice_id or voice_embedding must be specified.")

        if voice_id is not None and voice_embedding is not None:
            raise ValueError("Only one of voice_id or voice_embedding should be specified.")

        if voice_id:
            return {"mode": "id", "id": voice_id}

        return {"mode": "embedding", "embedding": voice_embedding}

    def send(
        self,
        model_id: str,
        transcript: str,
        output_format: dict,
        voice_id: Optional[str] = None,
        voice_embedding: Optional[List[float]] = None,
        context_id: Optional[str] = None,
        duration: Optional[int] = None,
        language: Optional[str] = None,
        stream: bool = True,
    ) -> Union[bytes, Generator[bytes, None, None]]:
        """Send a request to the WebSocket to generate audio.

        Args:
            model_id: The ID of the model to use for generating audio.
            transcript: The text to convert to speech.
            output_format: A dictionary containing the details of the output format.
            voice_id: The ID of the voice to use for generating audio.
            voice_embedding: The embedding of the voice to use for generating audio.
            context_id: The context ID to use for the request. If not specified, a random context ID will be generated.
            duration: The duration of the audio in seconds.
            language: The language code for the audio request. This can only be used with `model_id = sonic-multilingual`
            stream: Whether to stream the audio or not. (Default is True)

        Returns:
            If `stream` is True, the method returns a generator that yields chunks. Each chunk is a dictionary.
            If `stream` is False, the method returns a dictionary.
            Both the generator and the dictionary contain the following key(s):
            - audio: The audio as bytes.
            - context_id: The context ID for the request.
        """
        self.connect()

        if context_id is None:
            context_id = uuid.uuid4().hex

        voice = self._validate_and_construct_voice(voice_id, voice_embedding)

        request_body = {
            "model_id": model_id,
            "transcript": transcript,
            "voice": voice,
            "output_format": {
                "container": output_format["container"],
                "encoding": output_format["encoding"],
                "sample_rate": output_format["sample_rate"],
            },
            "context_id": context_id,
            "language": language,
        }

        if duration is not None:
            request_body["duration"] = duration

        generator = self._websocket_generator(request_body)

        if stream:
            return generator

        chunks = []
        for chunk in generator:
            chunks.append(chunk["audio"])

        return {"audio": b"".join(chunks), "context_id": context_id}

    def _websocket_generator(self, request_body: Dict[str, Any]):
        self.websocket.send(json.dumps(request_body))

        try:
            while True:
                response = json.loads(self.websocket.recv())
                if "error" in response:
                    raise RuntimeError(f"Error generating audio:\n{response['error']}")
                if response["done"]:
                    break
                yield self._convert_response(response=response, include_context_id=True)
        except Exception as e:
            # Close the websocket connection if an error occurs.
            if self.websocket and not self._is_websocket_closed():
                self.websocket.close()
            raise RuntimeError(f"Failed to generate audio. {response}") from e


class _SSE:
    """This class contains methods to generate audio using Server-Sent Events.

    Usage:
        >>> for audio_chunk in client.tts.sse(
        ...     model_id="upbeat-moon", transcript="Hello world!", voice_embedding=embedding,
        ...     output_format={"container": "raw", "encoding": "pcm_f32le", "sample_rate": 44100}, stream=True
        ... ):
        ...     audio = audio_chunk["audio"]
    """

    def __init__(
        self,
        http_url: str,
        headers: Dict[str, str],
        timeout: float,
    ):
        self.http_url = http_url
        self.headers = headers
        self.timeout = timeout

    def _update_buffer(self, buffer: str, chunk_bytes: bytes) -> Tuple[str, List[Dict[str, Any]]]:
        buffer += chunk_bytes.decode("utf-8")
        outputs = []
        while "{" in buffer and "}" in buffer:
            start_index = buffer.find("{")
            end_index = buffer.find("}", start_index)
            if start_index != -1 and end_index != -1:
                try:
                    chunk_json = json.loads(buffer[start_index : end_index + 1])
                    if "error" in chunk_json:
                        raise RuntimeError(f"Error generating audio:\n{chunk_json['error']}")
                    if chunk_json["done"]:
                        break
                    audio = base64.b64decode(chunk_json["data"])
                    outputs.append({"audio": audio})
                    buffer = buffer[end_index + 1 :]
                except json.JSONDecodeError:
                    break
        return buffer, outputs

    def _validate_and_construct_voice(
        self, voice_id: Optional[str] = None, voice_embedding: Optional[List[float]] = None
    ) -> dict:
        """Validate and construct the voice dictionary for the request.

        Args:
            voice_id: The ID of the voice to use for generating audio.
            voice_embedding: The embedding of the voice to use for generating audio.

        Returns:
            A dictionary representing the voice configuration.

        Raises:
            ValueError: If neither or both voice_id and voice_embedding are specified.
        """
        if voice_id is None and voice_embedding is None:
            raise ValueError("Either voice_id or voice_embedding must be specified.")

        if voice_id is not None and voice_embedding is not None:
            raise ValueError("Only one of voice_id or voice_embedding should be specified.")

        if voice_id:
            return {"mode": "id", "id": voice_id}

        return {"mode": "embedding", "embedding": voice_embedding}

    def send(
        self,
        model_id: str,
        transcript: str,
        output_format: OutputFormat,
        voice_id: Optional[str] = None,
        voice_embedding: Optional[List[float]] = None,
        duration: Optional[int] = None,
        language: Optional[str] = None,
        stream: bool = True,
    ) -> Union[bytes, Generator[bytes, None, None]]:
        """Send a request to the server to generate audio using Server-Sent Events.

        Args:
            model_id: The ID of the model to use for generating audio.
            transcript: The text to convert to speech.
            voice_id: The ID of the voice to use for generating audio.
            voice_embedding: The embedding of the voice to use for generating audio.
            output_format: A dictionary containing the details of the output format.
            duration: The duration of the audio in seconds.
            language: The language code for the audio request. This can only be used with `model_id = sonic-multilingual`
            stream: Whether to stream the audio or not.

        Returns:
            If `stream` is True, the method returns a generator that yields chunks. Each chunk is a dictionary.
            If `stream` is False, the method returns a dictionary.
            Both the generator and the dictionary contain the following key(s):
            - audio: The audio as bytes.
        """
        voice = self._validate_and_construct_voice(voice_id, voice_embedding)

        request_body = {
            "model_id": model_id,
            "transcript": transcript,
            "voice": voice,
            "output_format": {
                "container": output_format["container"],
                "encoding": output_format["encoding"],
                "sample_rate": output_format["sample_rate"],
            },
            "language": language,
        }

        if duration is not None:
            request_body["duration"] = duration

        generator = self._sse_generator_wrapper(request_body)

        if stream:
            return generator

        chunks = []
        for chunk in generator:
            chunks.append(chunk["audio"])

        return {"audio": b"".join(chunks)}

    @retry_on_connection_error(
        max_retries=MAX_RETRIES, backoff_factor=BACKOFF_FACTOR, logger=logger
    )
    def _sse_generator_wrapper(self, request_body: Dict[str, Any]):
        """Need to wrap the sse generator in a function for the retry decorator to work."""
        try:
            for chunk in self._sse_generator(request_body):
                yield chunk
        except Exception as e:
            logger.error(f"Failed to generate audio. {e}")
            raise e

    def _sse_generator(self, request_body: Dict[str, Any]):
        response = requests.post(
            f"{self.http_url}/tts/sse",
            stream=True,
            data=json.dumps(request_body),
            headers=self.headers,
            timeout=(self.timeout, self.timeout),
        )
        if not response.ok:
            raise ValueError(f"Failed to generate audio. {response.text}")

        buffer = ""
        for chunk_bytes in response.iter_content(chunk_size=None):
            buffer, outputs = self._update_buffer(buffer=buffer, chunk_bytes=chunk_bytes)
            for output in outputs:
                yield output

        if buffer:
            try:
                chunk_json = json.loads(buffer)
                audio = base64.b64decode(chunk_json["data"])
                yield {"audio": audio}
            except json.JSONDecodeError:
                pass


class TTS(Resource):
    """This resource contains methods to generate audio using Cartesia's text-to-speech API."""

    def __init__(self, api_key, timeout):
        super().__init__(
            api_key=api_key,
            timeout=timeout,
        )
        self._sse_class = _SSE(self._http_url(), self.headers, self.timeout)
        self.sse = self._sse_class.send

    def websocket(self) -> _WebSocket:
        """This method returns a WebSocket object that can be used to generate audio using WebSocket.

        Returns:
            _WebSocket: A WebSocket object that can be used to generate audio using WebSocket.
        """
        ws = _WebSocket(self._ws_url(), self.api_key, self.cartesia_version)
        ws.connect()
        return ws

    def get_output_format(self, output_format_name: str) -> OutputFormat:
        """Convenience method to get the output_format dictionary from a given output format name.

        Args:
            output_format_name (str): The name of the output format.

        Returns:
            OutputFormat: A dictionary containing the details of the output format to be passed into tts.sse() or tts.websocket().send()

        Raises:
            ValueError: If the output_format name is not supported
        """
        if output_format_name in OutputFormatMapping._format_mapping:
            output_format_obj = OutputFormatMapping.get_format(output_format_name)
        elif output_format_name in DeprecatedOutputFormatMapping._format_mapping:
            output_format_obj = DeprecatedOutputFormatMapping.get_format_deprecated(
                output_format_name
            )
        else:
            raise ValueError(f"Unsupported format: {output_format_name}")

        return OutputFormat(
            container=output_format_obj["container"],
            encoding=output_format_obj["encoding"],
            sample_rate=output_format_obj["sample_rate"],
        )

    def get_sample_rate(self, output_format_name: str) -> int:
        """Convenience method to get the sample rate for a given output format.

        Args:
            output_format_name (str): The name of the output format.

        Returns:
            int: The sample rate for the output format.

        Raises:
            ValueError: If the output_format name is not supported
        """
        if output_format_name in OutputFormatMapping._format_mapping:
            output_format_obj = OutputFormatMapping.get_format(output_format_name)
        elif output_format_name in DeprecatedOutputFormatMapping._format_mapping:
            output_format_obj = DeprecatedOutputFormatMapping.get_format_deprecated(
                output_format_name
            )
        else:
            raise ValueError(f"Unsupported format: {output_format_name}")

        return output_format_obj["sample_rate"]


class AsyncCartesia(Cartesia):
    """The asynchronous version of the Cartesia client."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_num_connections: int = DEFAULT_NUM_CONNECTIONS,
    ):
        """
        Args:
            api_key: See :class:`Cartesia`.
            timeout: See :class:`Cartesia`.
            max_num_connections: The maximum number of concurrent connections to use for the client.
                This is used to limit the number of connections that can be made to the server.
        """
        self._session = None
        self._loop = None
        super().__init__(api_key=api_key, timeout=timeout)
        self.max_num_connections = max_num_connections
        self.tts = AsyncTTS(
            api_key=self.api_key, timeout=self.timeout, get_session=self._get_session
        )

    async def _get_session(self):
        current_loop = asyncio.get_event_loop()
        if self._loop is not current_loop:
            # If the loop has changed, close the session and create a new one.
            await self.close()
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            connector = aiohttp.TCPConnector(limit=self.max_num_connections)
            self._session = aiohttp.ClientSession(timeout=timeout, connector=connector)
            self._loop = current_loop
        return self._session

    async def close(self):
        """This method closes the session.

        It is *strongly* recommended to call this method when you are done using the client.
        """
        if self._session is not None and not self._session.closed:
            await self._session.close()

    def __del__(self):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is None:
            asyncio.run(self.close())
        else:
            loop.create_task(self.close())

    async def __aenter__(self):
        return self

    async def __aexit__(
        self,
        exc_type: Union[type, None],
        exc: Union[BaseException, None],
        exc_tb: Union[TracebackType, None],
    ):
        await self.close()


class _AsyncSSE(_SSE):
    """This class contains methods to generate audio using Server-Sent Events asynchronously."""

    def __init__(
        self,
        http_url: str,
        headers: Dict[str, str],
        timeout: float,
        get_session: Callable[[], Optional[aiohttp.ClientSession]],
    ):
        super().__init__(http_url, headers, timeout)
        self._get_session = get_session

    async def send(
        self,
        model_id: str,
        transcript: str,
        output_format: OutputFormat,
        voice_id: Optional[str] = None,
        voice_embedding: Optional[List[float]] = None,
        duration: Optional[int] = None,
        language: Optional[str] = None,
        stream: bool = True,
    ) -> Union[bytes, AsyncGenerator[bytes, None]]:
        voice = self._validate_and_construct_voice(voice_id, voice_embedding)

        request_body = {
            "model_id": model_id,
            "transcript": transcript,
            "voice": voice,
            "output_format": {
                "container": output_format["container"],
                "encoding": output_format["encoding"],
                "sample_rate": output_format["sample_rate"],
            },
            "language": language,
        }

        if duration is not None:
            request_body["duration"] = duration

        generator = self._sse_generator_wrapper(request_body)

        if stream:
            return generator

        chunks = []
        async for chunk in generator:
            chunks.append(chunk["audio"])

        return {"audio": b"".join(chunks)}

    @retry_on_connection_error_async(
        max_retries=MAX_RETRIES, backoff_factor=BACKOFF_FACTOR, logger=logger
    )
    async def _sse_generator_wrapper(self, request_body: Dict[str, Any]):
        """Need to wrap the sse generator in a function for the retry decorator to work."""
        try:
            async for chunk in self._sse_generator(request_body):
                yield chunk
        except Exception as e:
            logger.error(f"Failed to generate audio. {e}")
            raise e

    async def _sse_generator(self, request_body: Dict[str, Any]):
        session = await self._get_session()
        async with session.post(
            f"{self.http_url}/tts/sse", data=json.dumps(request_body), headers=self.headers
        ) as response:
            if not response.ok:
                raise ValueError(f"Failed to generate audio. {await response.text()}")

            buffer = ""
            async for chunk_bytes in response.content.iter_any():
                buffer, outputs = self._update_buffer(buffer=buffer, chunk_bytes=chunk_bytes)
                for output in outputs:
                    yield output

            if buffer:
                try:
                    chunk_json = json.loads(buffer)
                    audio = base64.b64decode(chunk_json["data"])
                    yield {"audio": audio}
                except json.JSONDecodeError:
                    pass


class _AsyncWebSocket(_WebSocket):
    """This class contains methods to generate audio using WebSocket asynchronously."""

    def __init__(
        self,
        ws_url: str,
        api_key: str,
        cartesia_version: str,
        get_session: Callable[[], Optional[aiohttp.ClientSession]],
    ):
        super().__init__(ws_url, api_key, cartesia_version)
        self._get_session = get_session
        self.websocket = None

    async def connect(self):
        if self.websocket is None or self._is_websocket_closed():
            route = "tts/websocket"
            session = await self._get_session()
            self.websocket = await session.ws_connect(
                f"{self.ws_url}/{route}?api_key={self.api_key}&cartesia_version={self.cartesia_version}"
            )

    def _is_websocket_closed(self):
        return self.websocket.closed

    async def close(self):
        """This method closes the websocket connection. *Highly* recommended to call this method when done."""
        if self.websocket is not None and not self._is_websocket_closed():
            await self.websocket.close()

    async def send(
        self,
        model_id: str,
        transcript: str,
        output_format: OutputFormat,
        voice_id: Optional[str] = None,
        voice_embedding: Optional[List[float]] = None,
        context_id: Optional[str] = None,
        duration: Optional[int] = None,
        language: Optional[str] = None,
        stream: Optional[bool] = True,
    ) -> Union[bytes, AsyncGenerator[bytes, None]]:
        await self.connect()

        if context_id is None:
            context_id = uuid.uuid4().hex

        voice = self._validate_and_construct_voice(voice_id, voice_embedding)

        request_body = {
            "model_id": model_id,
            "transcript": transcript,
            "voice": voice,
            "output_format": {
                "container": output_format["container"],
                "encoding": output_format["encoding"],
                "sample_rate": output_format["sample_rate"],
            },
            "context_id": context_id,
            "language": language,
        }

        if duration is not None:
            request_body["duration"] = duration

        generator = self._websocket_generator(request_body)

        if stream:
            return generator

        chunks = []
        async for chunk in generator:
            chunks.append(chunk["audio"])

        return {"audio": b"".join(chunks), "context_id": context_id}

    async def _websocket_generator(self, request_body: Dict[str, Any]):
        await self.websocket.send_json(request_body)

        try:
            response = None
            while True:
                response = await self.websocket.receive_json()
                if "error" in response:
                    raise RuntimeError(f"Error generating audio:\n{response['error']}")
                if response["done"]:
                    break

                yield self._convert_response(response=response, include_context_id=True)
        except Exception as e:
            # Close the websocket connection if an error occurs.
            if self.websocket and not self._is_websocket_closed():
                await self.websocket.close()
            error_msg_end = "" if response is None else f": {await response.text()}"
            raise RuntimeError(f"Failed to generate audio. {error_msg_end}") from e


class AsyncTTS(TTS):
    def __init__(self, api_key, timeout, get_session):
        super().__init__(api_key, timeout)
        self._get_session = get_session
        self._sse_class = _AsyncSSE(self._http_url(), self.headers, self.timeout, get_session)
        self.sse = self._sse_class.send

    async def websocket(self) -> _AsyncWebSocket:
        ws = _AsyncWebSocket(self._ws_url(), self.api_key, self.cartesia_version, self._get_session)
        await ws.connect()
        return ws
