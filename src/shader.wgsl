// Documentation see: https://gpuweb.github.io/gpuweb/wgsl/

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
};

@vertex
fn vs_main(
    @builtin(vertex_index) in_vertex_index: u32,
) -> VertexOutput {
    var out: VertexOutput;
    let x = f32(1 - i32(in_vertex_index)) * 0.5;
    let y = f32(i32(in_vertex_index & 1u) * 2 - 1) * 0.5;
    out.clip_position = vec4<f32>(x, y, 0.0, 1.0);
    return out;
}

@fragment
fn frag_main(
    // @location(0) fragUV: vec2<f32>,
    // @location(1) fragPosition: vec4<f32>
    in: VertexOutput
) -> @location(0) vec4<f32> {
    // return vec4<f32>(fragPosition.x, fragPosition.y, fragPosition.z, 0.75);
    return vec4<f32>(in.clip_position.x + 0.1, in.clip_position.y + 0.1, in.clip_position.z+ 0.1, 1.0/*0.75*/);
}
