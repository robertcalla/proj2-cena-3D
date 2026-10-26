from __future__ import annotations

import math
from typing import Any

import graphics_math as gm

from light import Light


class SpotLight(Light):
    # posição, direção, abertura (meio-ângulo do cone, em graus) e expoente
    def __init__ (self, x: float, y: float, z: float,
                  dx: float, dy: float, dz: float,
                  angle: float, exponent: float,
                  space: str = "world", **values: Any) -> None:
      Light.__init__(self, space, **values)
      self.set_position(x, y, z)
      self.set_direction(dx, dy, dz)
      self.set_angle(angle)
      self.set_exponent(exponent)

    def set_position (self, x: float, y: float, z: float) -> None:
      self.set("light_position", gm.vec4(x,y,z,1.0))

    def set_direction (self, x: float, y: float, z: float) -> None:
      self.set("spot_direction", gm.vec3(x,y,z))

    def set_angle (self, angle: float) -> None:
      if not 0.0 < angle <= 180.0:
        raise ValueError("a abertura do spot deve estar em (0, 180] graus")
      # o shader só precisa do cosseno: calcula uma vez aqui, não por fragmento
      self.set("spot_cos_cutoff", math.cos(math.radians(angle)))

    def set_exponent (self, exponent: float) -> None:
      if exponent < 0.0:
        raise ValueError("o expoente do spot deve ser >= 0")
      self.set("spot_exponent", exponent)

    def map (self, matrix: gm.Mat4) -> dict[str, Any]:
      position = self.get("light_position")
      d = self.get("spot_direction")
      # direção ignora translação (w = 0) e volta normalizada
      direction = (matrix @ gm.vec4(d[0], d[1], d[2], 0.0))[:3]
      return {"light_position": matrix @ position,
              "spot_direction": gm.normalize(direction)}
