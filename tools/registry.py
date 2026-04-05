from .rag import RAGTool
from .time import TimeTool
from .distance import HaversineTool

TOOLS = {
    "rag_search": RAGTool(),
    "time": TimeTool(),
    "haversine": HaversineTool(),
}
