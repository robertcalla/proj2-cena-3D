// Projeto 2: Phong por fragmento com spot, decal, rugosidade, fog e plano de corte.
// Objetos sem decal usam textura branca; sem relevo, o mapa (0.5, 0.5, 1.0).

struct Matrix {
  vertex: mat4x4<f32>,
  normal: mat4x4<f32>,
}
@group(0) @binding(0) var<storage, read> matrix: array<Matrix>;

struct PhongBlock {
  base_color: vec3<f32>,
  opacity: f32,
  specular_color: vec3<f32>,
  shininess: f32,
}
@group(1) @binding(0) var<uniform> material: PhongBlock;

struct Global {
  projection: mat4x4<f32>,
  camera_position: vec4<f32>,
  light_position: vec4<f32>,
  light_ambient: vec3<f32>,
  light_diffuse: vec3<f32>,
  light_specular: vec3<f32>,
  spot_direction: vec3<f32>,
  spot_cos_cutoff: f32,
  spot_exponent: f32,
  global_ambient: vec3<f32>,
  fog_color: vec4<f32>,
  fog_density: f32,
  clip_plane: vec4<f32>,
  clip_plane_color: vec4<f32>,
  use_normal_map: f32,
}
@group(2) @binding(0) var<uniform> global: Global;

@group(3) @binding(0) var decal_texture: texture_2d<f32>;
@group(3) @binding(1) var decal_sampler: sampler;
@group(3) @binding(2) var normal_texture: texture_2d<f32>;
@group(3) @binding(3) var normal_sampler: sampler;

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) normal: vec3<f32>,
  @location(1) light: vec3<f32>,
  @location(2) view: vec3<f32>,
  @location(3) light_tan: vec3<f32>,
  @location(4) view_tan: vec3<f32>,
  @location(5) texcoord: vec2<f32>,
  @location(6) clip_dist: f32,
}

@vertex
fn vs_main (@builtin(instance_index) i: u32,
            @location(0) coord: vec3<f32>,
            @location(1) normal: vec3<f32>,
            @location(2) tangent: vec3<f32>,
            @location(3) texcoord: vec2<f32>) -> VertexOutput {
  let m = matrix[i];
  let p = (m.vertex * vec4<f32>(coord, 1.0)).xyz;
  let n = normalize((m.normal * vec4<f32>(normal, 0.0)).xyz);
  var t = normalize((m.normal * vec4<f32>(tangent, 0.0)).xyz);
  t = normalize(t - n * dot(n, t));
  let b = cross(n, t);
  // B^T: linhas t, b, n
  let to_tangent_space = mat3x3<f32>(
    vec3<f32>(t.x, b.x, n.x),
    vec3<f32>(t.y, b.y, n.y),
    vec3<f32>(t.z, b.z, n.z),
  );

  var out: VertexOutput;
  out.normal = n;
  out.light = global.light_position.xyz - p;
  out.view = global.camera_position.xyz - p;
  out.light_tan = to_tangent_space * out.light;
  out.view_tan = to_tangent_space * out.view;
  out.texcoord = texcoord;
  out.clip_dist = dot(global.clip_plane, vec4<f32>(p, 1.0));
  out.position = global.projection * vec4<f32>(p, 1.0);
  return out;
}

@fragment
fn fs_main (in: VertexOutput,
            @builtin(front_facing) is_front: bool) -> @location(0) vec4<f32> {
  // amostra antes de qualquer desvio
  let decal = textureSample(decal_texture, decal_sampler, in.texcoord);
  let texel = textureSample(normal_texture, normal_sampler, in.texcoord).rgb;

  if (in.clip_dist < 0.0) {
    discard;
  }

  let lw = normalize(in.light);
  var n: vec3<f32>;
  var l: vec3<f32>;
  var v: vec3<f32>;
  var matte: vec3<f32>;
  var macro_lit = true;
  if (!is_front) {
    // tampa do corte
    n = -normalize(global.clip_plane.xyz);
    l = lw;
    v = normalize(in.view);
    matte = global.clip_plane_color.rgb;
  } else if (global.use_normal_map > 0.5) {
    n = normalize(2.0 * texel - vec3<f32>(1.0));
    l = normalize(in.light_tan);
    v = normalize(in.view_tan);
    macro_lit = l.z > 0.0;
    matte = material.base_color * decal.rgb;
  } else {
    n = normalize(in.normal);
    l = lw;
    v = normalize(in.view);
    matte = material.base_color * decal.rgb;
  }

  let c = dot(-lw, global.spot_direction);
  var k_spot = 0.0;
  if (c >= global.spot_cos_cutoff) {
    k_spot = select(pow(max(c, 0.0), global.spot_exponent), 1.0,
                    global.spot_exponent == 0.0);
  }

  let ndotl = select(0.0, dot(n, l), macro_lit);
  var lit = matte * global.light_ambient
          + matte * global.light_diffuse * max(ndotl, 0.0);
  if (ndotl > 0.0) {
    let r = reflect(-l, n);
    lit += material.specular_color * global.light_specular
         * pow(max(dot(r, v), 0.0), material.shininess);
  }
  var color = matte * global.global_ambient + k_spot * lit;

  let s = length(in.view);
  let f = exp(-pow(global.fog_density * s, 2.0));
  color = f * color + (1.0 - f) * global.fog_color.rgb;
  return vec4<f32>(color, material.opacity);
}
