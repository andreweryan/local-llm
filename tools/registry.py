from .rag import RAGTool
from .distance import HaversineTool

TOOLS = {
    "rag_search": RAGTool(),
    "haversine": HaversineTool(),
}
