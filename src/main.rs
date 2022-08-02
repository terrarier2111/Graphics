mod api;

use std::mem::size_of;
use wgpu::{BlendState, Buffer, BufferAddress, BufferSlice, BufferUsages, ColorTargetState, ColorWrites, DeviceDescriptor, Face, Features, FrontFace, Limits, MultisampleState, PolygonMode, PresentMode, PrimitiveState, PrimitiveTopology, RenderPass, RenderPipeline, ShaderSource, SurfaceError, TextureUsages, vertex_attr_array, VertexAttribute, VertexBufferLayout, VertexStepMode};
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::WindowBuilder;
use crate::api::{FragmentShaderState, PipelineStateBuilder, RenderPassHandler, ShaderModuleSources, State, VertexShaderState};

fn main() {
    pollster::block_on(run());
}

async fn run() {
    // env_logger::init();
    let mut dynamic_color = true;
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new().with_title("Test Window").build(&event_loop).unwrap();
    let mut state = State::new(
        &window,
        Default::default(),
        &DeviceDescriptor {
            features: Features::empty(),
            limits: Limits::default(),
            label: None,
        },
        PresentMode::Fifo,
        TextureUsages::RENDER_ATTACHMENT,
    )
    .await
    .unwrap()
    .unwrap();
    state.setup_pipelines(vec![PipelineStateBuilder::new().bind_group_layouts(&[]).push_constant_ranges(&[]).vertex(VertexShaderState {
        entry_point: "vs_main",
        buffers: &[Vertex::desc()],
    }).fragment(FragmentShaderState {
        entry_point: "frag_main",
        targets: &[Some(ColorTargetState { // 4.
            format: state.format().clone(),
            blend: Some(BlendState::REPLACE),
            write_mask: ColorWrites::ALL,
        })],
    }).primitive(PrimitiveState {
        topology: PrimitiveTopology::TriangleList, // 1.
        strip_index_format: None,
        front_face: FrontFace::Ccw, // 2.
        cull_mode: Some(Face::Back),
        // Setting this to anything other than Fill requires Features::NON_FILL_POLYGON_MODE
        polygon_mode: PolygonMode::Fill,
        // Requires Features::DEPTH_CLIP_CONTROL
        unclipped_depth: false,
        // Requires Features::CONSERVATIVE_RASTERIZATION
        conservative: false,
    }).multisample(MultisampleState {
        count: 1, // 2.
        mask: !0, // 3.
        alpha_to_coverage_enabled: false, // 4.
    }).shader_src(ShaderModuleSources::Single(ShaderSource::Wgsl(include_str!("shader.wgsl").into()))).build(),
                               PipelineStateBuilder::new().bind_group_layouts(&[]).push_constant_ranges(&[]).vertex(VertexShaderState {
                                   entry_point: "vs_main",
                                   buffers: &[Vertex::desc()],
                               }).fragment(FragmentShaderState {
                                   entry_point: "frag_main",
                                   targets: &[Some(ColorTargetState { // 4.
                                       format: state.format().clone(),
                                       blend: Some(BlendState::REPLACE),
                                       write_mask: ColorWrites::ALL,
                                   })],
                               }).primitive(PrimitiveState {
                                   topology: PrimitiveTopology::TriangleList, // 1.
                                   strip_index_format: None,
                                   front_face: FrontFace::Ccw, // 2.
                                   cull_mode: Some(Face::Back),
                                   // Setting this to anything other than Fill requires Features::NON_FILL_POLYGON_MODE
                                   polygon_mode: PolygonMode::Fill,
                                   // Requires Features::DEPTH_CLIP_CONTROL
                                   unclipped_depth: false,
                                   // Requires Features::CONSERVATIVE_RASTERIZATION
                                   conservative: false,
                               }).multisample(MultisampleState {
                                   count: 1, // 2.
                                   mask: !0, // 3.
                                   alpha_to_coverage_enabled: false, // 4.
                               }).shader_src(ShaderModuleSources::Single(ShaderSource::Wgsl(include_str!("new_shader.wgsl").into()))).build()]);

    let vertex_buffer = state.create_buffer(&VERTICES, BufferUsages::VERTEX);

    let vertex_buffer_slice = vertex_buffer.slice(..);
    event_loop.run(move |event, _, control_flow| match event {
        Event::NewEvents(_) => {}
        Event::WindowEvent {
            ref event,
            window_id,
        } if window_id == window.id() => {
            if !state.input(event) {
                match event {
                    WindowEvent::Resized(size) => {
                        if !state.resize(*size) {
                            println!("Couldn't resize!");
                        }
                    }
                    WindowEvent::Moved(_) => {}
                    WindowEvent::CloseRequested => {
                        *control_flow = ControlFlow::Exit;
                    }
                    WindowEvent::Destroyed => {}
                    WindowEvent::DroppedFile(_) => {}
                    WindowEvent::HoveredFile(_) => {}
                    WindowEvent::HoveredFileCancelled => {}
                    WindowEvent::ReceivedCharacter(_) => {}
                    WindowEvent::Focused(_) => {}
                    WindowEvent::KeyboardInput { .. } => {}
                    WindowEvent::ModifiersChanged(_) => {}
                    WindowEvent::CursorMoved { .. } => {}
                    WindowEvent::CursorEntered { .. } => {}
                    WindowEvent::CursorLeft { .. } => {}
                    WindowEvent::MouseWheel { .. } => {}
                    WindowEvent::MouseInput { .. } => {}
                    WindowEvent::TouchpadPressure { .. } => {}
                    WindowEvent::AxisMotion { .. } => {}
                    WindowEvent::Touch(_) => {}
                    WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                        if !state.resize(**new_inner_size) {
                            println!("Couldn't resize!");
                        }
                    }
                    WindowEvent::ThemeChanged(_) => {}
                }
            }
        }
        Event::DeviceEvent { .. } => {}
        Event::UserEvent(_) => {}
        Event::Suspended => {}
        Event::Resumed => {}
        Event::MainEventsCleared => {
            // RedrawRequested will only trigger once, unless we manually
            // request it.
            window.request_redraw();
        }
        Event::RedrawRequested(window_id) if window_id == window.id() => {
            state.update();
            let dynamic_color = dynamic_color;
            match state.render(/*|mut render_pass, render_pipelines| {
                if dynamic_color {
                    // let val = state.render_pipeline(1).unwrap();
                    render_pass.set_pipeline(render_pipelines.get(1).unwrap()/*&val*/); // 2.
                    // state.try_set_render_pipeline(render_pass, 1);
                } else {
                    // let val = state.render_pipeline(0).unwrap();
                    render_pass.set_pipeline(render_pipelines.get(0).unwrap()/*&val*/); // 2.
                    // state.try_set_render_pipeline(render_pass, 0);
                }

                // render_pass.set_vertex_buffer(0, vertex_buffer_slice);

                // render_pass.draw(0..(VERTICES.len() as u32), 0..1); // 3.
            }*/SimpleRenderPassHandler {
                dynamic_color,
                vertex_buffer: vertex_buffer.slice(..),
            }) {
                Ok(_) => {}
                // Reconfigure the surface if lost
                Err(SurfaceError::Lost) => {
                    let size = state.size();
                    state.resize(size);
                }
                // The system is out of memory, we should probably quit
                Err(SurfaceError::OutOfMemory) => *control_flow = ControlFlow::Exit,
                // All other errors (Outdated, Timeout) should be resolved by the next frame
                Err(e) => eprintln!("{:?}", e),
            }
        }
        Event::RedrawEventsCleared => {}
        Event::LoopDestroyed => {}
        _ => {}
    });
}


#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],
    color: [f32; 4],
}

const VERTICES: &[Vertex] = &[
    Vertex { position: [0.0, 0.5, 0.0], color: [1.0, 0.0, 0.0, 1.0] },
    Vertex { position: [-0.5, -0.5, 0.0], color: [0.0, 1.0, 0.0, 1.0] },
    Vertex { position: [0.5, -0.5, 0.0], color: [0.0, 0.0, 1.0, 1.0] },
];

impl Vertex {
    const ATTRIBS: [VertexAttribute; 2] =
        vertex_attr_array![0 => Float32x3, 1 => Float32x4];

    fn desc<'a>() -> VertexBufferLayout<'a> {
        VertexBufferLayout {
            array_stride: size_of::<Self>() as BufferAddress,
            step_mode: VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }
}

struct SimpleRenderPassHandler<'a> {
    dynamic_color: bool,
    vertex_buffer: BufferSlice<'a>,
}

impl<'a> RenderPassHandler<'a> for SimpleRenderPassHandler<'a> {
    fn handle<'b: 'c, 'c>(&mut self, mut render_pass: RenderPass<'c>, render_pipelines: &'b Box<[RenderPipeline]>) where 'a: 'b {
        if self.dynamic_color {
            // let val = state.render_pipeline(1).unwrap();
            render_pass.set_pipeline(render_pipelines.get(1).unwrap()/*&val*/); // 2.
            // state.try_set_render_pipeline(render_pass, 1);
        } else {
            // let val = state.render_pipeline(0).unwrap();
            render_pass.set_pipeline(render_pipelines.get(0).unwrap()/*&val*/); // 2.
            // state.try_set_render_pipeline(render_pass, 0);
        }

        render_pass.set_vertex_buffer(0, self.vertex_buffer);

        render_pass.draw(0..(VERTICES.len() as u32), 0..1); // 3.
    }
}
