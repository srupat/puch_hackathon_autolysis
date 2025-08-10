import asyncio
from typing import Annotated
import os
from dotenv import load_dotenv
from fastmcp import FastMCP
from fastmcp.server.auth.providers.bearer import BearerAuthProvider, RSAKeyPair
from mcp import ErrorData, McpError
from mcp.server.auth.provider import AccessToken
from mcp.types import TextContent, ImageContent, INVALID_PARAMS, INTERNAL_ERROR
from pydantic import BaseModel, Field, AnyUrl
from io import StringIO

import httpx
import pandas as pd

# --- Load environment variables ---
load_dotenv()

TOKEN = os.environ.get("AUTH_TOKEN")
MY_NUMBER = os.environ.get("MY_NUMBER")

assert TOKEN is not None, "Please set AUTH_TOKEN in your .env file"
assert MY_NUMBER is not None, "Please set MY_NUMBER in your .env file"

# --- Auth Provider ---
class SimpleBearerAuthProvider(BearerAuthProvider):
    def __init__(self, token: str):
        k = RSAKeyPair.generate()
        super().__init__(public_key=k.public_key, jwks_uri=None, issuer=None, audience=None)
        self.token = token

    async def load_access_token(self, token: str) -> AccessToken | None:
        if token == self.token:
            return AccessToken(
                token=token,
                client_id="puch-client",
                scopes=["*"],
                expires_at=None,
            )
        return None

# --- Rich Tool Description model ---
class RichToolDescription(BaseModel):
    description: str
    use_when: str
    side_effects: str | None = None
    

    
class FetchCSV:
    USER_AGENT = "Puch/1.0 (Autonomous)"
    
    @staticmethod
    def convert_google_drive_link(url: str) -> str:
        """Convert Google Drive share link to direct download link."""
        if "drive.google.com" in url and "/file/d/" in url:
            file_id = url.split("/file/d/")[1].split("/")[0]
            return f"https://drive.google.com/uc?export=download&id={file_id}"
        return url  
    
    @classmethod
    async def fetch_csv(cls, url: str) -> pd.DataFrame:
        """Fetch CSV from public URL and return as DataFrame."""
        direct_url = cls.convert_google_drive_link(url)
        async with httpx.AsyncClient() as client:
            try:
                resp = await client.get(
                    direct_url,
                    follow_redirects=True,
                    headers={"User-Agent": cls.USER_AGENT},
                    timeout=30,
                )
            except httpx.HTTPError as e:
                raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to fetch CSV: {e!r}"))

            if resp.status_code >= 400:
                raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to fetch CSV - status code {resp.status_code}"))

        try:
            df = pd.read_csv(StringIO(resp.text))
        except Exception as e:
            raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to parse CSV: {e!r}"))

        return df
    
    @staticmethod
    def basic_analysis(df: pd.DataFrame) -> str:
        """Generate basic stats for the dataset."""
        buffer = []
        buffer.append(f"**Number of rows:** {len(df)}")
        buffer.append(f"**Number of columns:** {len(df.columns)}")
        buffer.append("\n**Column names:**")
        for col in df.columns:
            buffer.append(f"- {col} ({df[col].dtype})")
        buffer.append("\n**Summary statistics:**\n")
        buffer.append(df.describe(include="all").to_markdown())
        return "\n".join(buffer)
    
    @classmethod
    async def generate_report(cls, csv_url: str, user_goal: str) -> str:
        """Fetch CSV, run analysis, and return Markdown report."""
        df = await cls.fetch_csv(csv_url)
        basic_stats = cls.basic_analysis(df)

        return (
            f"# 📊 Dataset Analysis Report\n"
            f"**User Goal:** {user_goal}\n\n"
            f"## Basic Dataset Info\n"
            f"{basic_stats}\n\n"
            f"## Next Steps\n"
            f"- Explore columns relevant to your goal\n"
            f"- Check for missing values and clean data\n"
            f"- Run deeper analysis for trends or anomalies"
        )
        
mcp = FastMCP("Automatic Dataset Analyzer", auth=SimpleBearerAuthProvider(TOKEN))

@mcp.tool
async def validate() -> str:
    return MY_NUMBER

CSVAnalyzerDescription = RichToolDescription(
    description="Fetch a CSV file from a given URL, analyze it, and generate insights or next-step suggestions.",
    use_when="Use when the user provides a link to a CSV file and wants analysis of its contents.",
    side_effects="None — only reads the CSV file and returns a text report.",
)

@mcp.tool(description=CSVAnalyzerDescription.model_dump_json())
async def csv_analyzer(
    csv_url: Annotated[AnyUrl, Field(description="The URL of the CSV file")],
    user_goal: Annotated[str, Field(description="The analysis goal or intent for the dataset")]
) -> str:
    return await FetchCSV.generate_report(str(csv_url), user_goal)


async def main():
    print(f"🚀 Starting CSV analyzer MCP server on http://0.0.0.0:8086")
    await mcp.run_async("streamable-http", host="0.0.0.0", port=8086)

if __name__ == "__main__":
    asyncio.run(main())
