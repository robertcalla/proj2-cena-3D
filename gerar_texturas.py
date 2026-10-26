from __future__ import annotations

# Gera as texturas da cena em textures/
#   python gerar_texturas.py

import math
import os

import numpy as np
from PIL import Image

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "textures")


def save (name: str, rgb: np.ndarray) -> None:
  path = os.path.join(OUT, name)
  Image.fromarray(np.clip(rgb * 255.0 + 0.5, 0, 255).astype(np.uint8)).save(path)
  print("salvo", path)


def value_noise (h: int, w: int, cells: int, seed: int) -> np.ndarray:
  rng = np.random.default_rng(seed)
  g = rng.random((cells, cells))
  y = np.linspace(0, cells, h, endpoint=False)
  x = np.linspace(0, cells, w, endpoint=False)
  y0 = np.floor(y).astype(int); x0 = np.floor(x).astype(int)
  fy = y - y0; fx = x - x0
  fy = fy * fy * (3 - 2 * fy); fx = fx * fx * (3 - 2 * fx)
  y1 = (y0 + 1) % cells; x1 = (x0 + 1) % cells
  a = g[np.ix_(y0, x0)]; b = g[np.ix_(y0, x1)]
  c = g[np.ix_(y1, x0)]; d = g[np.ix_(y1, x1)]
  fx = fx[None, :]; fy = fy[:, None]
  return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


def madeira () -> None:
  h = w = 512
  y, x = np.mgrid[0:h, 0:w] / h
  n = value_noise(h, w, 8, 1) + 0.5 * value_noise(h, w, 32, 2)
  r = np.sqrt((y - 1.7) ** 2 * 60.0 + (x - 0.5) ** 2 * 0.6) + 0.35 * n
  t = (0.5 + 0.5 * np.sin(r * 2 * math.pi * 3.0)) ** 3
  claro = np.array([0.86, 0.66, 0.42])
  escuro = np.array([0.55, 0.34, 0.17])
  fibra = 0.06 * (value_noise(h, w, 128, 3) - 0.5)
  rgb = claro * (1 - t[..., None]) + escuro * t[..., None] + fibra[..., None]
  save("madeira.png", rgb)
  # pernas: veios na vertical
  save("madeira_vertical.png", np.transpose(rgb, (1, 0, 2)))


def presente () -> None:
  h = w = 256
  y, x = (np.mgrid[0:h, 0:w] + 0.5) / h
  rgb = np.empty((h, w, 3))
  rgb[:] = (0.95, 0.78, 0.22)
  gx = (x * 8) % 1 - 0.5
  gy = (y * 8 + 0.5 * (np.floor(x * 8) % 2)) % 1 - 0.5
  rgb[gx ** 2 + gy ** 2 < 0.13 ** 2] = (1.0, 0.97, 0.85)
  for coord in (x, y):
    d = np.abs(coord - 0.5)
    rgb[d < 0.085] = (0.80, 0.10, 0.12)
    rgb[(d >= 0.085) & (d < 0.10)] = (0.55, 0.05, 0.07)
  save("presente.png", rgb)


def piso () -> None:
  # 8 x 8 ladrilhos, um por unidade do piso
  h = w = 1024
  y, x = (np.mgrid[0:h, 0:w] + 0.5) / h * 8
  n = value_noise(h, w, 16, 4)
  par = (np.floor(x) + np.floor(y)) % 2
  base = np.where(par[..., None] > 0, (0.62, 0.60, 0.56), (0.48, 0.46, 0.43))
  rgb = base * (0.9 + 0.2 * n[..., None])
  fx = np.abs(x - np.round(x)); fy = np.abs(y - np.round(y))
  rgb[(fx < 0.02) | (fy < 0.02)] = (0.30, 0.29, 0.27)
  save("piso.png", rgb)


def bola_praia () -> None:
  h, w = 256, 512
  y, x = np.mgrid[0:h, 0:w] + 0.5
  y, x = y / h, x / w
  cores = np.array([(0.90, 0.12, 0.12), (1.0, 1.0, 1.0), (0.10, 0.35, 0.85),
                    (1.0, 1.0, 1.0), (0.98, 0.80, 0.10), (1.0, 1.0, 1.0)])
  gomo = np.floor(x * len(cores)).astype(int) % len(cores)
  rgb = cores[gomo]
  rgb[(y < 0.08) | (y > 0.92)] = (1.0, 1.0, 1.0)
  save("bola_praia.png", rgb)


def fibonacci_sphere (n: int) -> np.ndarray:
  i = np.arange(n) + 0.5
  phi = np.arccos(1 - 2 * i / n)
  theta = math.pi * (1 + 5 ** 0.5) * i
  return np.stack([np.cos(theta) * np.sin(phi), np.cos(phi), np.sin(theta) * np.sin(phi)], axis=1)


def golfe_normal () -> None:
  # altura calculada sobre a esfera (mesma parametrizacao de sphere.py)
  # para as covinhas nao deformarem perto dos polos
  h, w = 512, 1024
  s = (np.arange(w) + 0.5) / w
  t = (np.arange(h) + 0.5) / h
  theta = s[None, :] * 2 * math.pi
  phi = (1.0 - t[:, None]) * math.pi
  p = np.stack([np.sin(theta) * np.sin(math.pi - phi) * np.ones_like(phi),
                np.cos(math.pi - phi) * np.ones_like(theta),
                np.cos(theta) * np.sin(math.pi - phi) * np.ones_like(phi)], axis=-1)

  radius = 0.088
  height = np.zeros((h, w))
  for c in fibonacci_sphere(320):
    d2 = np.sum((p - c) ** 2, axis=-1)
    inside = d2 < radius * radius
    height[inside] = np.minimum(height[inside], -(1.0 - d2[inside] / (radius * radius)))

  # ps = [2 0 dh/ds], pt = [0 2 dh/dt], n = ps x pt
  strength = 3.2
  hs = (np.roll(height, -1, axis=1) - np.roll(height, 1, axis=1)) * strength
  hp = np.pad(height, ((1, 1), (0, 0)), mode="edge")
  ht = (hp[:-2, :] - hp[2:, :]) * strength
  n = np.stack([-2.0 * hs, -2.0 * ht, np.full_like(hs, 4.0)], axis=-1)
  n /= np.linalg.norm(n, axis=-1, keepdims=True)
  save("golfe_normal.png", 0.5 * n + 0.5)


def main () -> None:
  os.makedirs(OUT, exist_ok=True)
  madeira()
  presente()
  piso()
  bola_praia()
  golfe_normal()


if __name__ == "__main__":
  main()
