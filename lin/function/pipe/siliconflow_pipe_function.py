"""
SiliconFlow Pipe - API Integration for Open WebUI

title: SiliconFlow Pipe
authors: linsuisheng034
author_url: none
funding_url: none
version: 0.4.0
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
    SiliconFlow API integration pipe for Open WebUI.
    Supports multiple models and handles text inputs.
    """
    
    class Valves(BaseModel):
        """
        Configuration options for the SiliconFlow pipe.
        """
        API_BASE_URL: str = Field(
            default="https://api.siliconflow.cn/v1",
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
        self.id = "siliconflow"
        self.name = "siliconflow/"
        self.valves = self.Valves(
            **{"API_KEY": os.getenv("SILICONFLOW_API_KEY", "")}
        )
        logger.info("SiliconFlow pipe initialized")

    def pipes(self) -> List[Dict[str, str]]:
        """
        Returns available models from the SiliconFlow API.
        
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
            if "siliconflowbycline." in model_name:
                model_name = model_name.replace("siliconflowbycline.", "")
            # Use original model name without prefix
            body["model"] = model_name
            
            # Validate input
            if not body.get("messages"):
                raise ValueError("No messages provided")
                
            system_message, messages = pop_system_message(body["messages"])
            
            # 从原始请求中获取所有参数，移除不支持的 tools 参数
            payload = {k: v for k, v in body.items() if k != "tools"}
            
            # 确保基本参数存在
            payload.update({
                "model": model_name,
                "messages": messages,
                "stream": body.get("stream", False)
            })

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
            # 确保 headers 包含正确的 accept 类型
            headers.update({"accept": "application/json"})
            
            # 确保 payload 中设置 stream 为 True
            payload["stream"] = True
            
            with requests.post(
                url,
                headers=headers,
                json=payload,
                stream=True,  # 请求时设置 stream 为 True
                timeout=(3.05, self.valves.TIMEOUT)
            ) as response:
                if response.status_code != 200:
                    raise Exception(
                        f"HTTP Error {response.status_code}: {response.text}"
                    )

                # 使用 iter_content 而不是 iter_lines
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        # 直接解码并返回文本内容
                        text = chunk.decode('utf-8')
                        yield text
                        
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