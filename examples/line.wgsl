struct Uniforms {
    view_proj: mat4x4<f32>,
    dash_length: f32,
}
@group(0) @binding(0) var<uniform> u: Uniforms;

struct VIn {
    @location(0) pos: vec3f,
    @location(1) color: vec4f,
    @location(2) arc_len: f32,
}

struct VOut {
    @builtin(position) pos: vec4f,
    @location(0) color: vec4f,
    @location(1) arc_len: f32,
}

@vertex fn vs(in: VIn) -> VOut {
    var out: VOut;
    out.pos = u.view_proj * vec4f(in.pos, 1.0);
    out.color = in.color;
    out.arc_len = in.arc_len;
    return out;
}

@fragment fn fs(in: VOut) -> @location(0) vec4f {
    // dash_length <= 0 means solid line
    if u.dash_length > 0.0 {
        let seg = u32(floor(in.arc_len / u.dash_length));
        if seg % 2u == 1u {
            discard;
        }
    }
    return in.color;
}
