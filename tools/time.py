from datetime import datetime
from .base import Tool
import tzlocal


class TimeTool(Tool):
    name = "time"
    description = "Return the current local time including DST. Defaults to UTC if local timezone cannot be determined."

    def run(self, query, app):
        """
        Returns the current local time as ISO 8601, DST-aware.
        """
        try:
            # tzlocal gives the system timezone with DST
            local_tz = tzlocal.get_localzone()
        except Exception:
            # fallback to UTC
            import zoneinfo

            local_tz = zoneinfo.ZoneInfo("UTC")

        now = datetime.now(local_tz)
        return now.isoformat(), []
