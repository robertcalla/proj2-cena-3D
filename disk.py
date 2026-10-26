from __future__ import annotations

from typing import TYPE_CHECKING

import math
import numpy as np
import wgpu
from shape import Shape

if TYPE_CHECKING:
  from state import State

class Disk (Shape):
  # disco unitário (raio 1, centro na origem), em leque de triângulos
  def __init__ (self, device: wgpu.GPUDevice, nslices: int = 32) -> None:
    if nslices < 3:
      raise ValueError("Disk needs at least 3 slices")
    coord = [[0.0, 0.0]]
    texcoord = [[0.5, 0.5]]
    for i in range(nslices):
      theta = 2 * math.pi * i / nslices
      x, y = math.cos(theta), math.sin(theta)
      coord.append([x, y])
      texcoord.append([0.5 + 0.5 * x, 0.5 - 0.5 * y])  # t cresce pra baixo (webgpu)
    bcoord = np.array(coord, dtype='float32')
    btexcoord = np.array(texcoord, dtype='float32')

    index: list[int] = []
    for i in range(nslices):
      j = i + 1
      k = (i + 1) % nslices + 1
      index += [0, j, k]
    bindex = np.array(index, dtype='uint32')
    self.nind: int = len(index)

    self.coord_vbo: wgpu.GPUBuffer = device.create_buffer_with_data(data=bcoord, usage=wgpu.BufferUsage.VERTEX)
    self.texcoord_vbo: wgpu.GPUBuffer = device.create_buffer_with_data(data=btexcoord, usage=wgpu.BufferUsage.VERTEX)
    self.ibo: wgpu.GPUBuffer = device.create_buffer_with_data(data=bindex, usage=wgpu.BufferUsage.INDEX)

  def draw (self, st: State) -> None:
    first_instance = st.get_shader().commit_matrix(st)
    st.render_pass.set_vertex_buffer(0, self.coord_vbo)
    st.render_pass.set_vertex_buffer(1, self.texcoord_vbo)
    st.render_pass.set_index_buffer(self.ibo, wgpu.IndexFormat.uint32)
    st.render_pass.draw_indexed(self.nind, 1, 0, 0, first_instance)
