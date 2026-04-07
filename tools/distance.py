import re
from .base import Tool
from math import radians, sin, cos, asin, sqrt


class HaversineTool(Tool):
    name = "haversine"
    description = "Calculate the Haversine distance between two geographic coordinates from a natural language prompt."
    direct_response = True

    def run(self, query: str, app, **kwargs):
        """
        Accepts a prompt like:
        "calculate the distance in kilometers between (lat1, lon1) and (lat2, lon2)"
        """
        # Default unit
        unit = "kilometers"

        # Detect if the user specified the unit in the text
        unit_match = re.search(r"in (\w+)", query.lower())
        if unit_match:
            unit_candidate = unit_match.group(1)
            if unit_candidate in ("kilometers", "km", "miles", "meters", "feet"):
                unit = (
                    "kilometers"
                    if unit_candidate in ("km", "kilometers")
                    else unit_candidate
                )

        # Extract coordinates
        coord_matches = re.findall(r"\(([-\d.]+),\s*([-\d.]+)\)", query)
        if len(coord_matches) != 2:
            return "Error: Could not parse two coordinate pairs from your query.", []

        lat1, lon1 = map(float, coord_matches[0])
        lat2, lon2 = map(float, coord_matches[1])

        # Haversine calculation
        earth_radius = {
            "kilometers": 6371.009,
            "meters": 6371009,
            "miles": 3958.7614581,
            "feet": 20902260.49876800925,
        }

        r = earth_radius[unit]
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        h = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        h = min(1, h)
        arc = 2 * asin(sqrt(h))
        distance = arc * r

        return f"{distance:.3f} {unit}", []
