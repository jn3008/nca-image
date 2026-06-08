export const updateShader = /* wgsl */ `
struct SimParams {
  width: u32,
  height: u32,
  tick: u32,
  mode: u32,
};

@group(0) @binding(0) var<storage, read> inputState: array<f32>;
@group(0) @binding(1) var<storage, read_write> outputState: array<f32>;
@group(0) @binding(2) var<storage, read> w1: array<f32>;
@group(0) @binding(3) var<storage, read> b1: array<f32>;
@group(0) @binding(4) var<storage, read> w2: array<f32>;
@group(0) @binding(5) var<uniform> params: SimParams;

const CHANNELS: u32 = 16u;
const PERCEPTION: u32 = 48u;
const HIDDEN: u32 = 128u;

fn cellIndex(x: u32, y: u32, c: u32) -> u32 {
  return ((y * params.width + x) * CHANNELS) + c;
}

fn sampleState(xi: i32, yi: i32, c: u32) -> f32 {
  if (xi < 0 || yi < 0 || xi >= i32(params.width) || yi >= i32(params.height)) {
    return 0.0;
  }
  return inputState[cellIndex(u32(xi), u32(yi), c)];
}

fn aliveAt(xi: i32, yi: i32) -> bool {
  for (var oy = -1; oy <= 1; oy = oy + 1) {
    for (var ox = -1; ox <= 1; ox = ox + 1) {
      if (sampleState(xi + ox, yi + oy, 3u) > 0.1) {
        return true;
      }
    }
  }
  return false;
}

fn rand01(x: u32, y: u32, tick: u32) -> f32 {
  var n = x * 1973u + y * 9277u + tick * 26699u + 911u;
  n = (n << 13u) ^ n;
  let nn = n * (n * n * 15731u + 789221u) + 1376312589u;
  return f32(nn & 65535u) / 65535.0;
}

fn sobelX(x: i32, y: i32, c: u32) -> f32 {
  return (
    -sampleState(x - 1, y - 1, c) + sampleState(x + 1, y - 1, c) +
    -2.0 * sampleState(x - 1, y, c) + 2.0 * sampleState(x + 1, y, c) +
    -sampleState(x - 1, y + 1, c) + sampleState(x + 1, y + 1, c)
  ) / 8.0;
}

fn sobelY(x: i32, y: i32, c: u32) -> f32 {
  return (
    -sampleState(x - 1, y - 1, c) - 2.0 * sampleState(x, y - 1, c) - sampleState(x + 1, y - 1, c) +
    sampleState(x - 1, y + 1, c) + 2.0 * sampleState(x, y + 1, c) + sampleState(x + 1, y + 1, c)
  ) / 8.0;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let x = id.x;
  let y = id.y;
  if (x >= params.width || y >= params.height) {
    return;
  }

  let xi = i32(x);
  let yi = i32(y);
  let preAlive = aliveAt(xi, yi);
  var next: array<f32, 16>;

  if (params.mode == 0u) {
    for (var c = 0u; c < CHANNELS; c = c + 1u) {
      let center = sampleState(xi, yi, c);
      let diffusion = (
        sampleState(xi - 1, yi, c) + sampleState(xi + 1, yi, c) +
        sampleState(xi, yi - 1, c) + sampleState(xi, yi + 1, c)
      ) * 0.25 - center;
      next[c] = center + diffusion * 0.12;
    }
    let pulse = max(0.0, 1.0 - abs(f32(params.tick % 160u) - 80.0) / 80.0);
    next[0] = max(next[0] * 0.996, next[3] * 0.10);
    next[1] = max(next[1] * 0.996, next[3] * (0.45 + 0.20 * pulse));
    next[2] = max(next[2] * 0.996, next[3] * 0.80);
    next[3] = clamp(next[3] * 0.998 + (sampleState(xi, yi, 4u) * 0.002), 0.0, 1.0);
    next[4] = clamp(next[4] * 0.998 + next[3] * 0.003, 0.0, 1.0);
  } else {
    var p: array<f32, 48>;
    for (var c = 0u; c < CHANNELS; c = c + 1u) {
      let base = c * 3u;
      p[base] = sampleState(xi, yi, c);
      p[base + 1u] = sobelX(xi, yi, c);
      p[base + 2u] = sobelY(xi, yi, c);
    }

    var h: array<f32, 128>;
    for (var j = 0u; j < HIDDEN; j = j + 1u) {
      var sum = b1[j];
      for (var i = 0u; i < PERCEPTION; i = i + 1u) {
        sum = sum + p[i] * w1[j * PERCEPTION + i];
      }
      h[j] = max(sum, 0.0);
    }

    for (var c = 0u; c < CHANNELS; c = c + 1u) {
      var ds = 0.0;
      for (var j = 0u; j < HIDDEN; j = j + 1u) {
        ds = ds + h[j] * w2[c * HIDDEN + j];
      }
      let stochastic = select(0.0, 1.0, rand01(x, y, params.tick) <= 0.5);
      next[c] = sampleState(xi, yi, c) + ds * stochastic;
    }
  }

  var postAlive = false;
  if (next[3] > 0.1) {
    postAlive = true;
  }
  for (var oy = -1; oy <= 1; oy = oy + 1) {
    for (var ox = -1; ox <= 1; ox = ox + 1) {
      if (sampleState(xi + ox, yi + oy, 3u) > 0.1) {
        postAlive = true;
      }
    }
  }

  let alive = preAlive && postAlive;
  for (var c = 0u; c < CHANNELS; c = c + 1u) {
    outputState[cellIndex(x, y, c)] = select(0.0, next[c], alive);
  }
}
`;

export const renderShader = /* wgsl */ `
struct SimParams {
  width: u32,
  height: u32,
  tick: u32,
  mode: u32,
};

struct VertexOut {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

@group(0) @binding(0) var<storage, read> state: array<f32>;
@group(0) @binding(1) var<uniform> params: SimParams;

const CHANNELS: u32 = 16u;

fn cellIndex(x: u32, y: u32, c: u32) -> u32 {
  return ((y * params.width + x) * CHANNELS) + c;
}

@vertex
fn vs(@builtin(vertex_index) vertexIndex: u32) -> VertexOut {
  var positions = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>(3.0, -1.0),
    vec2<f32>(-1.0, 3.0)
  );
  let pos = positions[vertexIndex];
  var out: VertexOut;
  out.position = vec4<f32>(pos, 0.0, 1.0);
  out.uv = pos * 0.5 + vec2<f32>(0.5);
  return out;
}

@fragment
fn fs(input: VertexOut) -> @location(0) vec4<f32> {
  let gx = min(u32(input.uv.x * f32(params.width)), params.width - 1u);
  let gy = min(u32((1.0 - input.uv.y) * f32(params.height)), params.height - 1u);
  let r = clamp(state[cellIndex(gx, gy, 0u)], 0.0, 1.0);
  let g = clamp(state[cellIndex(gx, gy, 1u)], 0.0, 1.0);
  let b = clamp(state[cellIndex(gx, gy, 2u)], 0.0, 1.0);
  let a = clamp(state[cellIndex(gx, gy, 3u)], 0.0, 1.0);
  let bg = vec3<f32>(1.0);
  let rgb = vec3<f32>(r, g, b) * a + bg * (1.0 - a);
  return vec4<f32>(rgb, 1.0);
}
`;
