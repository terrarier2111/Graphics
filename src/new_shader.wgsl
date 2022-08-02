// Documentation see: https://gpuweb.github.io/gpuweb/wgsl/

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn vs_main(
    // @builtin(vertex_index) in_vertex_index: u32,
    model: VertexInput,
) -> VertexOutput {
    var out: VertexOutput;
    // let x = f32(1 - i32(in_vertex_index)) * 0.5;
    // let y = f32(i32(in_vertex_index & 1u) * 2 - 1) * 0.5;
    // out.clip_position = vec4<f32>(x, y, 0.0, 1.0);
    out.clip_position = vec4<f32>(model.position, 1.0);
    out.color = model.color/* * vec4<f32>(cos(model.position.x * 16.0), cos(model.position.y * 16.0), 0.0, 1.0)*/;
    return out;
}

@fragment
fn frag_main(
    // @location(0) fragUV: vec2<f32>,
    // @location(1) fragPosition: vec4<f32>
    in: VertexOutput
) -> @location(0) vec4<f32> {
    return in.color;
}
