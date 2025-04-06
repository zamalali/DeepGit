import httpx
import logging

logger = logging.getLogger(__name__)

class MCPAdapter:
    def __init__(self):
        self.adapter_name = "GitHub MCP Adapter"
        # Optionally, initialize shared client settings or cache here.

    async def fetch(self, url: str, headers: dict = None, params: dict = None, client: httpx.AsyncClient = None):
        """
        A standardized fetch method that wraps HTTP GET calls.
        If a client is provided, it uses it; otherwise, it creates a temporary client.
        """
        try:
            if client is None:
                async with httpx.AsyncClient() as temp_client:
                    response = await temp_client.get(url, headers=headers, params=params)
            else:
                response = await client.get(url, headers=headers, params=params)
            logger.info(f"[{self.adapter_name}] Fetched URL: {url} with status {response.status_code}")
            return response
        except Exception as e:
            logger.error(f"[{self.adapter_name}] Error fetching {url}: {e}")
            raise e

# Provide a singleton instance for use in other modules.
mcp_adapter = MCPAdapter()
