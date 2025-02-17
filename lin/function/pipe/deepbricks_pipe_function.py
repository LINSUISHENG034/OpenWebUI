"""
Deepbricks Pipe - API Integration for Open WebUI

title: Deepbricks Pipe
authors: linsuisheng034
author_url: none
funding_url: none
version: 0.2.0
required_open_webui_version: 0.3.17
license: MIT
"""

import os
import requests
import json
import time
import logging
from typing import List, Union, Generator, Iterator, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from open_webui.utils.misc import pop_system_message
from fastapi import Request

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Pipe:
    """
    Deepbricks API integration pipe for Open WebUI.
    Supports multiple models and handles both text and image inputs.
    """
    
    class Valves(BaseModel):
        """
        Configuration options for the Deepbricks pipe.
        """
        API_BASE_URL: str = Field(
            default="https://api.deepbricks.ai/v1",
            description="Base URL for the API endpoint"
        )
        API_KEY: str = Field(
            default="",
            description="API key for authentication"
        )
        TIMEOUT: int = Field(
            default=60,
            description="Timeout in seconds for API requests"
        )
        
        @validator('API_KEY')
        def validate_api_key(cls, v):
            if not v:
                logger.warning("API key is not set")
            return v

    def __init__(self):
        """Initialize the pipe with default values"""
        self.type = "manifold"
        self.id = "deepbricks"
        self.name = "deepbricks/"
        self.valves = self.Valves(
            **{"API_KEY": os.getenv("DEEPBRICKS_API_KEY", "")}
        )
        self.MAX_IMAGE_SIZE = 5 * 1024 * 1024  # 5MB per image
        self.MAX_TOTAL_SIZE = 100 * 1024 * 1024  # 100MB total
        logger.info("Deepbricks pipe initialized")

    def pipes(self) -> List[Dict[str, str]]:
        """
        Returns available models from the Deepbricks API.
        
        Returns:
            List of dictionaries containing model IDs and names
        """
        if not self.valves.API_KEY:
            logger.error("API key not provided")
            return [{"id": "error", "name": "API Key not provided"}]
            
        try:
            headers = {
                "Authorization": f"Bearer {self.valves.API_KEY}",
                "Content-Type": "application/json"
            }
            
            response = requests.get(
                f"{self.valves.API_BASE_URL}/models",
                headers=headers,
                timeout=self.valves.TIMEOUT
            )
            response.raise_for_status()
            
            models = response.json()
            return [
                {
                    "id": model["id"],
                    "name": model.get('name', model['id'])
                }
                for model in models.get("data", [])
                if model.get("id")
            ]
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch models: {str(e)}")
            return [{"id": "error", "name": "Error fetching models"}]

    def process_image(self, image_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process image data with size validation and format conversion.
        
        Args:
            image_data: Dictionary containing image URL or base64 data
            
        Returns:
            Dictionary with processed image data
            
        Raises:
            ValueError: If image size exceeds limits or format is invalid
        """
        try:
            if image_data["image_url"]["url"].startswith("data:image"):
                mime_type, base64_data = image_data["image_url"]["url"].split(",", 1)
                media_type = mime_type.split(":")[1].split(";")[0]

                # Check base64 image size
                image_size = len(base64_data) * 3 / 4
                if image_size > self.MAX_IMAGE_SIZE:
                    raise ValueError(
                        f"Image size exceeds 5MB limit: {image_size / (1024 * 1024):.2f}MB"
                    )

                return {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": base64_data,
                    },
                }
            else:
                # For URL images, perform size check
                url = image_data["image_url"]["url"]
                response = requests.head(url, allow_redirects=True)
                content_length = int(response.headers.get("content-length", 0))

                if content_length > self.MAX_IMAGE_SIZE:
                    raise ValueError(
                        f"Image at URL exceeds 5MB limit: {content_length / (1024 * 1024):.2f}MB"
                    )

                return {
                    "type": "image",
                    "source": {"type": "url", "url": url},
                }
                
        except Exception as e:
            logger.error(f"Failed to process image: {str(e)}")
            raise ValueError(f"Image processing error: {str(e)}")

    async def pipe(self, body: dict, __user__: dict, __request__: Optional[Request] = None) -> Union[str, Generator, Iterator]:
        """
        Main processing function for the pipe.
        
        Args:
            body: Input data containing messages and parameters
            __user__: User information dictionary
            __request__: Optional FastAPI request object
            
        Returns:
            Processed response from the API
            
        Raises:
            ValueError: If input validation fails
        """
        try:
            # Format model name
            model_name = body["model"]
            if "deepbricksbycline." in model_name:
                model_name = model_name.replace("deepbricksbycline.", "")
            # Use original model name without prefix
            body["model"] = model_name
            
            # Validate input
            if not body.get("messages"):
                raise ValueError("No messages provided")
                
            system_message, messages = pop_system_message(body["messages"])
            processed_messages = []
            total_image_size = 0

            # Process each message
            for message in messages:
                processed_content = []
                if isinstance(message.get("content"), list):
                    for item in message["content"]:
                        if item["type"] == "text":
                            processed_content.append({"type": "text", "text": item["text"]})
                        elif item["type"] == "image_url":
                            processed_image = self.process_image(item)
                            processed_content.append(processed_image)

                            # Track total size for base64 images
                            if processed_image["source"]["type"] == "base64":
                                image_size = len(processed_image["source"]["data"]) * 3 / 4
                                total_image_size += image_size
                                if total_image_size > self.MAX_TOTAL_SIZE:
                                    raise ValueError(
                                        f"Total size of images exceeds {self.MAX_TOTAL_SIZE / (1024 * 1024)}MB limit"
                                    )
                else:
                    processed_content = [
                        {"type": "text", "text": message.get("content", "")}
                    ]

                # Convert content to string if role is assistant
                if message["role"] == "assistant":
                    content_str = ""
                    for item in processed_content:
                        if item["type"] == "text":
                            content_str += item["text"] + "\n"
                    processed_messages.append(
                        {"role": message["role"], "content": content_str.strip()}
                    )
                else:
                    processed_messages.append(
                        {"role": message["role"], "content": processed_content}
                    )

            # Validate and prepare model name
            model_name = body["model"]
            if not model_name:
                raise ValueError("Model name is required")
                
            # Prepare payload
            payload = {
                "model": model_name,
                "messages": processed_messages,
                "max_tokens": body.get("max_tokens", 4096),
                "temperature": body.get("temperature", 0.8),
                "stream": body.get("stream", False),
            }

            headers = {
                "Authorization": f"Bearer {self.valves.API_KEY}",
                "Content-Type": "application/json"
            }

            endpoint = f"{self.valves.API_BASE_URL}/chat/completions"

            if body.get("stream", False):
                return self.stream_response(endpoint, headers, payload)
            else:
                return await self.non_stream_response(endpoint, headers, payload)
                
        except ValueError as e:
            logger.error(f"Validation error: {str(e)}")
            return f"Validation error: {str(e)}"
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {str(e)}")
            return f"Request failed: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return f"Unexpected error: {str(e)}"

    def stream_response(self, url: str, headers: dict, payload: dict) -> Generator:
        """
        Handle streaming responses from the API.
        
        Args:
            url: API endpoint URL
            headers: Request headers
            payload: Request payload
            
        Yields:
            Chunks of response data
            
        Raises:
            Exception: If API request fails
        """
        try:
            with requests.post(
                url,
                headers=headers,
                json=payload,
                stream=True,
                timeout=(3.05, self.valves.TIMEOUT)
            ) as response:
                if response.status_code != 200:
                    raise Exception(
                        f"HTTP Error {response.status_code}: {response.text}"
                    )

                for line in response.iter_lines():
                    if line:
                        line = line.decode("utf-8")
                        if line.startswith("data: "):
                            try:
                                data = json.loads(line[6:])
                                yield data["choices"][0]["delta"].get("content", "")
                                time.sleep(0.01)  # Delay to avoid overwhelming client
                            except json.JSONDecodeError:
                                logger.error(f"Failed to parse JSON: {line}")
                            except KeyError as e:
                                logger.error(f"Unexpected data structure: {e}")
                                logger.debug(f"Full data: {data}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Stream request failed: {str(e)}")
            yield f"Error: Stream request failed: {str(e)}"
        except Exception as e:
            logger.error(f"Stream processing error: {str(e)}")
            yield f"Error: {str(e)}"

    async def non_stream_response(self, url: str, headers: dict, payload: dict) -> str:
        """
        Handle non-streaming responses from the API.
        
        Args:
            url: API endpoint URL
            headers: Request headers
            payload: Request payload
            
        Returns:
            Processed response content
            
        Raises:
            Exception: If API request fails
        """
        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=(3.05, self.valves.TIMEOUT)
            )
            response.raise_for_status()
            
            res = response.json()
            return res["choices"][0]["message"]["content"]
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Non-stream request failed: {str(e)}")
            return f"Error: Non-stream request failed: {str(e)}"
        except KeyError as e:
            logger.error(f"Unexpected response format: {str(e)}")
            return f"Error: Unexpected response format"
        except Exception as e:
            logger.error(f"Unexpected error in non-stream response: {str(e)}")
            return f"Error: {str(e)}"
