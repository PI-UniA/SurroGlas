class Temperature_profile:
    def __init__(self, x_polynom: list[float] = None, y_polynom: list[float] = None):
        self.x_polynom = x_polynom
        self.y_polynom = y_polynom


class Zones:
    def __init__(
        self,
        name: str = None,
        temp_profile: Temperature_profile = Temperature_profile(),
        zone_length: float = None,
    ):
        self.name = name
        self.temp_profile = temp_profile
        self.zone_length = zone_length


class IncomingDto:
    def __init__(
        self,
        number_of_zones: int = None,
        lehrspeed: int = None,
        glass_thickness: int = None,
        glass_width: int = None,
        zone_width: int = None,
        zone_dimension: list[Zones] = None,
    ):
        self.number_of_zones = number_of_zones
        self.lehrspeed = lehrspeed
        self.glass_thickness = glass_thickness
        self.glass_width = glass_width
        self.zone_width = zone_width
        self.zone_dimension = zone_dimension
