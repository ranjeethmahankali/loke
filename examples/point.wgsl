struct Uniforms {
    view_proj: mat4x4<f32>,
    dash_length: f32,
    point_size: f32,
    screen_width: f32,
    screen_height: f32,
}
@group(0) @binding(0) var<uniform> u: Uniforms;

struct PointIn {
    @location(0) center: vec3f,
    @location(1) color: vec4f,
}

struct VOut {
    @builtin(position) pos: vec4f,
    @location(0) color: vec4f,
    @location(1) uv: vec2f,
}

// 6 vertices per quad: two triangles
const QUAD_UVS = array<vec2f, 6>(
    vec2f(-1.0, -1.0),
    vec2f( 1.0, -1.0),
    vec2f( 1.0,  1.0),
    vec2f(-1.0, -1.0),
    vec2f( 1.0,  1.0),
    vec2f(-1.0,  1.0),
);

@vertex fn vs(
    point: PointIn,
    @builtin(vertex_index) vi: u32,
) -> VOut {
    let clip_center = u.view_proj * vec4f(point.center, 1.0);
    let uv = QUAD_UVS[vi];
    // Offset in clip space by point_size pixels
    let pixel = vec2f(2.0 / u.screen_width, 2.0 / u.screen_height);
    var out: VOut;
    out.pos = clip_center + vec4f(uv * pixel * u.point_size * clip_center.w, 0.0, 0.0);
    out.color = point.color;
    out.uv = uv;
    return out;
}

@fragment fn fs(in: VOut) -> @location(0) vec4f {
    let r2 = dot(in.uv, in.uv);
    if r2 > 1.0 {
        discard;
    }
    // Implicit sphere shading: z = sqrt(1 - r²), use as normal.z
    let nz = sqrt(1.0 - r2);
    // Simple directional light from upper-right-front
    let light = normalize(vec3f(0.3, 0.5, 1.0));
    let normal = vec3f(in.uv, nz);
    let diffuse = max(dot(normal, light), 0.0);
    let ambient = 0.25;
    let shade = ambient + (1.0 - ambient) * diffuse;
    return vec4f(in.color.rgb * shade, in.color.a);
}
