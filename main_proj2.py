# INF1761 - Projeto 2 opcao 1: mesa com objetos

from __future__ import annotations
import math
import os
import time
from typing import Any

import graphics_math as gm
import wgpu

from camera3d import *
from spotlight import *
from shader import *
from pipeline import *
from phongmaterial import *
from transform import *
from node import *
from scene import *
from renderer import *
from cube import *
from sphere import *
from texture import *
from sampler import *
from textureset import *

HERE = os.path.dirname(os.path.abspath(__file__))
SHADER = os.path.join(HERE, "shaders", "cena.wgsl")
TEXTURES = os.path.join(HERE, "textures")

EYE = (1.7, 3.3, 4.0)
CENTER = (0.05, 1.1, 0.0)
FOVY = 40.0

CLEAR_COLOR = (0.80, 0.84, 0.90, 1.0)

SPOT_POS = (-0.6, 3.4, 2.2)
SPOT_TARGET = (0.0, 1.0, -0.2)
SPOT_ANGLE = 24.0
SPOT_EXP = 12.0
GLOBAL_AMBIENT = (0.12, 0.12, 0.14)

FOG_DENSITY = 0.11
CLIP_NORMAL = (-1.0, 0.0, 0.0)
CLIP_D = -0.25
CLIP_COLOR = (0.85, 0.25, 0.20)

TABLE_W = 2.4
TABLE_D = 1.8
TOP_T = 0.12
LEG = 0.14
LEG_H = 1.1
FLOOR = 8.0


def srgb (r: float, g: float, b: float) -> tuple[float, float, float]:
  # cores escolhidas na tela, convertidas para linear
  def lin (c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
  return (lin(r), lin(g), lin(b))


def phong (rgb: tuple[float, float, float], spec: float, shininess: float) -> PhongMaterial:
  mat = PhongMaterial(*srgb(*rgb))
  mat.set_specular(spec, spec, spec)
  mat.set_shininess(shininess)
  return mat


def box (sx: float, sy: float, sz: float, tx: float = 0.0, ty: float = 0.0, tz: float = 0.0) -> Transform:
  trf = Transform()
  trf.translate(tx, ty, tz)
  trf.scale(sx, sy, sz)
  return trf


def ball (radius: float, tx: float, ty: float, tz: float) -> Transform:
  trf = Transform()
  trf.translate(tx, ty, tz)
  trf.scale(radius, radius, radius)
  return trf


class Effects:
  # o piso usa outro shader, onde o plano de corte fica sempre desligado

  def __init__ (self, shader: Shader, floor_shader: Shader) -> None:
    self.shader = shader
    self.floor_shader = floor_shader
    self.fog = True
    self.clip = False
    self.clip_d = CLIP_D
    self.animate_clip = False
    self.bump = True
    for shd in (shader, floor_shader):
      shd.set_value("fog_color", gm.vec4(*CLEAR_COLOR))
      shd.set_value("clip_plane_color", gm.vec4(*srgb(*CLIP_COLOR), 1.0))
    floor_shader.set_value("clip_plane", gm.vec4(0.0, 0.0, 0.0, 1.0))
    self.apply()

  def apply (self) -> None:
    for shd in (self.shader, self.floor_shader):
      shd.set_value("fog_density", FOG_DENSITY if self.fog else 0.0)
      shd.set_value("use_normal_map", 1.0 if self.bump else 0.0)
    if self.clip:
      self.shader.set_value("clip_plane", gm.vec4(*CLIP_NORMAL, self.clip_d))
    else:
      self.shader.set_value("clip_plane", gm.vec4(0.0, 0.0, 0.0, 1.0))

  def update (self, t: float) -> None:
    if self.animate_clip:
      self.clip_d = 1.1 * math.sin(0.8 * t)
      self.apply()


def build_scene (device: wgpu.GPUDevice, target_format: str) -> tuple[Scene, Camera3D, Effects]:
  camera = Camera3D(*EYE)
  camera.set_center(*CENTER)
  camera.set_angle(FOVY)

  sdir = [t - p for t, p in zip(SPOT_TARGET, SPOT_POS)]
  light = SpotLight(*SPOT_POS, *sdir, SPOT_ANGLE, SPOT_EXP, space="world")
  light.set_ambient(0.05, 0.05, 0.05)
  light.set_diffuse(1.0, 1.0, 1.0)
  light.set_specular(1.0, 1.0, 1.0)

  def make_shader () -> Shader:
    shader = Shader(device, SHADER, light=light, space="world")
    shader.set_value("global_ambient", gm.vec3(*GLOBAL_AMBIENT))
    shader.set_vertex_buffers([
      {"array_stride": 3 * 4, "step_mode": "vertex",
       "attributes": [{"format": "float32x3", "offset": 0, "var_name": "coord"}]},
      {"array_stride": 3 * 4, "step_mode": "vertex",
       "attributes": [{"format": "float32x3", "offset": 0, "var_name": "normal"}]},
      {"array_stride": 3 * 4, "step_mode": "vertex",
       "attributes": [{"format": "float32x3", "offset": 0, "var_name": "tangent"}]},
      {"array_stride": 2 * 4, "step_mode": "vertex",
       "attributes": [{"format": "float32x2", "offset": 0, "var_name": "texcoord"}]},
    ])
    return shader

  shader = make_shader()
  floor_shader = make_shader()
  effects = Effects(shader, floor_shader)
  # sem cull: a tampa do corte usa as faces de tras
  pipeline = Pipeline(shader, target_format, primitive={"cull_mode": "none"})
  floor_pipeline = Pipeline(floor_shader, target_format, primitive={"cull_mode": "back"})

  def tex (name: str) -> Texture:
    return Texture(device, "decal_texture", os.path.join(TEXTURES, name))
  decal_sampler = Sampler(device, "decal_sampler")
  normal_sampler = Sampler(device, "normal_sampler")
  white = Texture(device, "decal_texture", None, texel=gm.vec3(1.0, 1.0, 1.0))
  # mapa de normais e dado, nao cor: rgba8unorm
  flat = Texture(device, "normal_texture", None, texel=gm.vec3(0.5, 0.5, 1.0), format="rgba8unorm")
  golf = Texture(device, "normal_texture", os.path.join(TEXTURES, "golfe_normal.png"), format="rgba8unorm")

  def texset (shd: Shader, decal: Texture, nmap: Texture = flat) -> TextureSet:
    ts = TextureSet([decal, decal_sampler, nmap, normal_sampler])
    shd.add_texture_set(ts)
    return ts

  ts_wood = texset(shader, tex("madeira.png"))
  ts_wood_vertical = texset(shader, tex("madeira_vertical.png"))
  ts_gift = texset(shader, tex("presente.png"))
  ts_beach = texset(shader, tex("bola_praia.png"))
  ts_golf = texset(shader, white, golf)
  ts_floor = texset(floor_shader, tex("piso.png"))

  wood = phong((1.0, 1.0, 1.0), 0.25, 16.0)
  gift = phong((1.0, 1.0, 1.0), 0.35, 32.0)
  beach = phong((1.0, 1.0, 1.0), 0.8, 64.0)
  golf_mat = phong((0.62, 0.86, 0.52), 0.7, 48.0)
  for mat in (wood, gift, beach, golf_mat):
    shader.add_material(mat)
  tile = phong((1.0, 1.0, 1.0), 0.15, 8.0)
  floor_shader.add_material(tile)

  cube = Cube(device)
  sphere = Sphere(device)

  floor = Node(floor_pipeline, nodes=[
    Node(trf=box(FLOOR, 0.05, FLOOR, 0.0, -0.05, 0.0), apps=[tile, ts_floor], shps=[cube]),
  ])

  lx = TABLE_W / 2 - LEG / 2
  lz = TABLE_D / 2 - LEG / 2
  legs = [Node(trf=box(LEG, LEG_H, LEG, sx * lx, 0.0, sz * lz), apps=[ts_wood_vertical], shps=[cube])
          for sx in (-1, 1) for sz in (-1, 1)]
  top = Node(trf=box(TABLE_W, TOP_T, TABLE_D, 0.0, LEG_H, 0.0), shps=[cube])

  BOX_W, BOX_H, BOX_D = 0.80, 0.28, 0.60
  R_GOLF = 0.17
  R_BEACH = 0.32

  # caixa um pouco acima do tampo para evitar z-fighting na tampa do corte
  box_trf = Transform()
  box_trf.translate(-0.25, 0.003, 0.05)
  box_trf.rotate(-8.0, 0.0, 1.0, 0.0)
  golf_ball = Node(trf=ball(R_GOLF, 0.0, BOX_H + R_GOLF, 0.0), apps=[golf_mat, ts_golf], shps=[sphere])
  box_node = Node(trf=box_trf, nodes=[
    Node(trf=box(BOX_W, BOX_H, BOX_D), apps=[gift, ts_gift], shps=[cube]),
    golf_ball,
  ])

  beach_trf = ball(R_BEACH, TABLE_W / 2 - 0.42, R_BEACH, -TABLE_D / 2 + 0.40)
  beach_trf.rotate(25.0, 0.0, 1.0, 0.0)
  beach_ball = Node(trf=beach_trf, apps=[beach, ts_beach], shps=[sphere])

  # origem na face de cima do tampo
  surface_trf = Transform()
  surface_trf.translate(0.0, LEG_H + TOP_T, 0.0)
  surface = Node(trf=surface_trf, nodes=[box_node, beach_ball])

  table = Node(apps=[wood, ts_wood], nodes=[top, *legs, surface])

  root = Node(nodes=[floor, Node(pipeline, nodes=[table])])
  return Scene(root), camera, effects


canvas: Any
device: wgpu.GPUDevice
context: Any
renderer: Renderer
camera: Camera3D
scene: Scene
effects: Effects
t0 = time.perf_counter()

_drag: dict[str, Any] = {"button": 0, "x": 0.0, "y": 0.0}


def attach_manipulator (canvas: Any, camera: Camera3D) -> None:
  arcball = camera.create_arcball()

  def on_down (event: Any) -> None:
    w, h = canvas.get_logical_size()
    _drag["button"] = event["button"]
    _drag["x"], _drag["y"] = event["x"], h - event["y"]
    if event["button"] == 1:
      arcball.init_mouse_motion(event["x"], h - event["y"])

  def on_up (event: Any) -> None:
    _drag["button"] = 0

  def on_move (event: Any) -> None:
    w, h = canvas.get_logical_size()
    x, y = event["x"], h - event["y"]
    if _drag["button"] == 1:
      arcball.accumulate_mouse_motion(x, y, w, h)
    elif _drag["button"] == 2:
      k = 2.0 * math.tan(gm.radians(camera.get_angle()) / 2.0) / h
      arcball.translate((x - _drag["x"]) * k, (y - _drag["y"]) * k, 0.0)
      _drag["x"], _drag["y"] = x, y

  def on_wheel (event: Any) -> None:
    fovy = camera.get_angle() * math.exp(0.001 * event["dy"])
    camera.set_angle(min(max(fovy, 8.0), 100.0))

  canvas.add_event_handler(on_down, "pointer_down")
  canvas.add_event_handler(on_up, "pointer_up")
  canvas.add_event_handler(on_move, "pointer_move")
  canvas.add_event_handler(on_wheel, "wheel")


def update_title () -> None:
  on = lambda flag: "on" if flag else "off"
  anim = " (animado)" if effects.animate_clip else ""
  canvas.set_title(f"projeto 2 | fog [f]: {on(effects.fog)} | corte [c]: {on(effects.clip)}{anim}"
                   f" | rugosidade [b]: {on(effects.bump)}")


def draw () -> None:
  effects.update(time.perf_counter() - t0)
  target_texture = context.get_current_texture()
  renderer.render(target_texture, scene, camera)


def on_key (event: Any) -> None:
  key = event["key"]
  if key == "q":
    canvas.close()
    return
  if key == "r":
    camera.get_arcball().mat = gm.mat4(1.0)
    camera.set_angle(FOVY)
  elif key == "f":
    effects.fog = not effects.fog
  elif key == "c":
    effects.clip = not effects.clip
    if not effects.clip:
      effects.animate_clip = False
  elif key == "p":
    effects.animate_clip = not effects.animate_clip
    if effects.animate_clip:
      effects.clip = True
  elif key in ("ArrowUp", "ArrowDown"):
    effects.clip = True
    effects.animate_clip = False
    effects.clip_d += 0.05 if key == "ArrowUp" else -0.05
  elif key == "b":
    effects.bump = not effects.bump
  effects.apply()
  update_title()


def main () -> None:
  global canvas, device, context, renderer, scene, camera, effects
  from rendercanvas.glfw import RenderCanvas, loop

  canvas = RenderCanvas(size=(900, 700), title="projeto 2",
                        update_mode="continuous", max_fps=60)
  adapter = wgpu.gpu.request_adapter_sync()
  device = adapter.request_device_sync()
  context = canvas.get_context("wgpu")
  target_format = context.get_preferred_format(device.adapter)
  context.configure(device=device, format=target_format)

  renderer = Renderer(device, depth_test=True, clear_value=CLEAR_COLOR)
  scene, camera, effects = build_scene(device, target_format)
  attach_manipulator(canvas, camera)
  update_title()

  canvas.add_event_handler(on_key, "key_down")
  canvas.request_draw(draw)
  loop.run()


if __name__ == "__main__":
  main()
