struct Uniforms {
    view_proj: mat4x4<f32>,
    dash_length: f32,
    point_size: f32,
    screen_width: f32,
    screen_height: f32,
}
@group(0) @binding(0) var<uniform> u: Uniforms;

struct AxisIn {
    @location(0) start: vec3f,
    @location(1) end: vec3f,
    @location(2) color: vec4f,
}

struct VOut {
    @builtin(position) pos: vec4f,
    @location(0) color: vec4f,
}

const WIDTH: f32 = 3.0; // pixels half-width

@vertex fn vs(
    axis: AxisIn,
    @builtin(vertex_index) vi: u32,
) -> VOut {
    let clip_a = u.view_proj * vec4f(axis.start, 1.0);
    let clip_b = u.view_proj * vec4f(axis.end, 1.0);
    // Screen-space direction
    let ndc_a = clip_a.xy / clip_a.w;
    let ndc_b = clip_b.xy / clip_b.w;
    let screen_a = ndc_a * vec2f(u.screen_width, u.screen_height) * 0.5;
    let screen_b = ndc_b * vec2f(u.screen_width, u.screen_height) * 0.5;
    let dir = normalize(screen_b - screen_a);
    let perp = vec2f(-dir.y, dir.x);
    // 6 verts: quad from start to end
    var along: f32;
    var side: f32;
    switch vi {
        case 0u: { along = 0.0; side = -1.0; }
        case 1u: { along = 0.0; side =  1.0; }
        case 2u: { along = 1.0; side =  1.0; }
        case 3u: { along = 0.0; side = -1.0; }
        case 4u: { along = 1.0; side =  1.0; }
        case 5u: { along = 1.0; side = -1.0; }
        default: { along = 0.0; side = 0.0; }
    }
    let clip_pos = mix(clip_a, clip_b, along);
    let pixel = vec2f(2.0 / u.screen_width, 2.0 / u.screen_height);
    let offset = perp * side * WIDTH * pixel * clip_pos.w;
    var out: VOut;
    out.pos = clip_pos + vec4f(offset, 0.0, 0.0);
    out.color = axis.color;
    return out;
}

@fragment fn fs(in: VOut) -> @location(0) vec4f {
    return in.color;
}
