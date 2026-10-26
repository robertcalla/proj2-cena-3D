from __future__ import annotations

# Gera as figuras do relatorio sem abrir janela
#   python render_figuras.py [nome ...]

import os
import sys
from typing import Callable

import numpy as np
import wgpu
from PIL import Image

import graphics_math as gm
from renderer import Renderer
from main_proj2 import build_scene, CLEAR_COLOR, FOVY, Effects

W, H = 900, 700
FORMAT = "rgba8unorm-srgb"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "relatorio", "figuras")


def drag (arcball, x0: float, y0: float, x1: float, y1: float, steps: int = 20) -> None:
  arcball.init_mouse_motion(x0, y0)
  for k in range(1, steps + 1):
    t = k / steps
    arcball.accumulate_mouse_motion(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, W, H)


def snapshot (device, renderer, scene, camera, path: str) -> None:
  tex = device.create_texture(size=(W, H, 1), format=FORMAT,
                              usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC)
  renderer.render(tex, scene, camera)
  data = device.queue.read_texture({"texture": tex}, {"bytes_per_row": W * 4, "rows_per_image": H}, (W, H, 1))
  img = np.frombuffer(data, dtype=np.uint8).reshape(H, W, 4)
  Image.fromarray(img[:, :, :3]).save(path)
  print("salvo", os.path.relpath(path))


def effects (fog: bool = False, clip: bool = False, bump: bool = True, d: float | None = None) -> Callable[[Effects], None]:
  def setup (fx: Effects) -> None:
    fx.fog, fx.clip, fx.bump = fog, clip, bump
    if d is not None:
      fx.clip_d = d
    fx.apply()
  return setup


def main () -> None:
  os.makedirs(OUT, exist_ok=True)
  adapter = wgpu.gpu.request_adapter_sync()
  device = adapter.request_device_sync()
  renderer = Renderer(device, depth_test=True, clear_value=CLEAR_COLOR)
  cx, cy = W / 2, H / 2

  none = lambda a, c: None
  # zoom na bola de golfe
  closeup = lambda a, c: (c.set_center(-0.22, 1.50, 0.05), c.set_angle(15.0))
  views = {
    "basica": (effects(bump=False), none),
    "giro_horizontal": (effects(bump=False), lambda a, c: drag(a, cx - 90, cy, cx + 90, cy)),
    "vista_de_cima": (effects(bump=False), lambda a, c: drag(a, cx, cy + 90, cx, cy - 40)),
    "fog_off": (effects(fog=False), lambda a, c: c.set_angle(55.0)),
    "fog_on": (effects(fog=True), lambda a, c: c.set_angle(55.0)),
    "corte": (effects(clip=True), none),
    "corte_proximo": (effects(clip=True, d=-0.15), closeup),
    "rugosidade_off": (effects(bump=False), closeup),
    "rugosidade_on": (effects(bump=True), closeup),
    "completa": (effects(fog=True, clip=True, bump=True), lambda a, c: drag(a, cx - 40, cy, cx + 30, cy + 10)),
  }
  only = set(sys.argv[1:])
  for name, (setup_fx, setup_view) in views.items():
    if only and name not in only:
      continue
    scene, camera, fx = build_scene(device, FORMAT)
    setup_fx(fx)
    arcball = camera.create_arcball()
    arcball.mat = gm.mat4(1.0)
    camera.set_angle(FOVY)
    setup_view(arcball, camera)
    snapshot(device, renderer, scene, camera, os.path.join(OUT, name + ".png"))


if __name__ == "__main__":
  main()
