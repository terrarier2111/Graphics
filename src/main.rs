#![feature(new_uninit)]

mod api;
mod model;
mod resources;

use crate::api::{
    FragmentShaderState, PipelineBuilder, ShaderModuleSources, State, StateBuilder, TextureBuilder,
    VertexShaderState,
};
use crate::model::{DrawModel, Model, ModelVertex, Vertex};
use cgmath::prelude::*;
use cgmath::{perspective, Deg, Matrix4, Point3, Quaternion, Vector3};
use parking_lot::Mutex;
use rand::Rng;
use std::mem::size_of;
use std::sync::Arc;
use std::thread;
use std::thread::sleep;
use std::time::Duration;
use wgpu::{
    AddressMode, BindGroupEntry, BindGroupLayoutEntry, BindingResource, BindingType, BlendState,
    Buffer, BufferAddress, BufferBindingType, BufferUsages, Color, ColorTargetState, ColorWrites,
    CompareFunction, DepthStencilState, Face, FilterMode, FrontFace, LoadOp, MultisampleState,
    Operations, PolygonMode, PrimitiveState, PrimitiveTopology, RenderPassColorAttachment,
    RenderPassDepthStencilAttachment, SamplerBindingType, SamplerDescriptor, ShaderSource,
    ShaderStages, SurfaceError, TextureDimension, TextureFormat, TextureSampleType,
    TextureViewDescriptor, TextureViewDimension, VertexAttribute, VertexBufferLayout, VertexFormat,
    VertexStepMode,
};
use winit::event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::WindowBuilder;

const DEPTH_FORMAT: TextureFormat = TextureFormat::Depth32Float;

fn main() {
    pollster::block_on(run());
}

async fn run() {
    let dynamic_color = true;
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("Test Window")
        .build(&event_loop)
        .unwrap();
    let state = StateBuilder::new().window(&window).build().await.unwrap();

    const SPACE_BETWEEN: f32 = 3.0;
    let instances = (0..NUM_INSTANCES_PER_ROW)
        .flat_map(|z| {
            (0..NUM_INSTANCES_PER_ROW).map(move |x| {
                let x = SPACE_BETWEEN * (x as f32 - NUM_INSTANCES_PER_ROW as f32 / 2.0);
                let z = SPACE_BETWEEN * (z as f32 - NUM_INSTANCES_PER_ROW as f32 / 2.0);

                let position = Vector3 { x, y: 0.0, z };

                let rotation = if position.is_zero() {
                    Quaternion::from_axis_angle(Vector3::unit_z(), Deg(0.0))
                } else {
                    Quaternion::from_axis_angle(position.normalize(), Deg(45.0))
                };

                Instance { position, rotation }
            })
        })
        .collect::<Vec<_>>();

    let instance_data = instances.iter().map(Instance::to_raw).collect::<Vec<_>>();
    // acos
    let camera = Camera {
        // position the camera one unit up and 2 units back
        // +z is out of the screen
        eye: (0.0, 1.0, 2.0).into(),
        // have it look at the origin
        target: (0.0, 0.0, 0.0).into(),
        // which way is "up"
        up: Vector3::unit_y(),
        aspect: state.size().0 as f32 / state.size().0 as f32,
        fovy: 45.0,
        znear: 0.1,
        zfar: 100.0,
    };
    let mut camera_uniform = CameraUniform::new();
    camera_uniform.update_view_proj(&camera);
    let camera_controller = CameraController::new(0.2);
    let camera_buffer = state.create_buffer(
        &[camera_uniform],
        BufferUsages::UNIFORM | BufferUsages::COPY_DST,
    );

    let instance_buffer = Arc::new(state.create_buffer(
        &instance_data,
        BufferUsages::VERTEX | BufferUsages::COPY_DST,
    ));

    // let vertex_buffer = state.create_buffer(&VERTICES, BufferUsages::VERTEX);
    // let index_buffer = state.create_buffer(&INDICES, BufferUsages::INDEX);
    let tree_tex = {
        let diffuse_bytes = include_bytes!("happy-tree.png");
        let diffuse_image = image::load_from_memory(diffuse_bytes).unwrap();
        let diffuse_rgba = diffuse_image.to_rgba8();

        use image::GenericImageView;
        let dimensions = diffuse_image.dimensions();
        // TEXTURE_BINDING tells wgpu that we want to use this texture in shaders
        state.create_texture(
            TextureBuilder::new()
                .data(&diffuse_rgba)
                .dimensions(dimensions)
                .format(TextureFormat::Rgba8UnormSrgb)
                .texture_dimension(TextureDimension::D2),
        )
    };
    // We don't need to configure the texture view much, so let's
    // let wgpu define it.
    let tree_texture_view = tree_tex.create_view(&TextureViewDescriptor::default());
    let tree_sampler = state.device().create_sampler(&SamplerDescriptor {
        address_mode_u: AddressMode::ClampToEdge,
        address_mode_v: AddressMode::ClampToEdge,
        address_mode_w: AddressMode::ClampToEdge,
        mag_filter: FilterMode::Linear,
        min_filter: FilterMode::Nearest,
        mipmap_filter: FilterMode::Nearest,
        ..Default::default()
    });
    let bind_group_layout = state.create_bind_group_layout(&[
        BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::FRAGMENT,
            ty: BindingType::Texture {
                multisampled: false,
                view_dimension: TextureViewDimension::D2,
                sample_type: TextureSampleType::Float { filterable: true },
            },
            count: None,
        },
        BindGroupLayoutEntry {
            binding: 1,
            visibility: ShaderStages::FRAGMENT,
            // This should match the filterable field of the
            // corresponding Texture entry above.
            ty: BindingType::Sampler(SamplerBindingType::Filtering),
            count: None,
        },
    ]);
    let obj_model = Model::load_from("cube.obj", &state, &bind_group_layout)
        .await
        .unwrap();
    let bind_group = Arc::new(Mutex::from(state.create_bind_group(
        &bind_group_layout,
        &[
            BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureView(&tree_texture_view),
            },
            BindGroupEntry {
                binding: 1,
                resource: BindingResource::Sampler(&tree_sampler),
            },
        ],
    )));
    let camera_bind_group_layout = state.create_bind_group_layout(&[BindGroupLayoutEntry {
        binding: 0,
        visibility: ShaderStages::VERTEX,
        ty: BindingType::Buffer {
            ty: BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }]);
    let camera_bind_group = state.create_bind_group(
        &camera_bind_group_layout,
        &[BindGroupEntry {
            binding: 0,
            resource: camera_buffer.as_entire_binding(),
        }],
    );
    let pipelines =
        [
            state.create_pipeline(
                PipelineBuilder::new()
                    .layout(&state.create_pipeline_layout(
                        &[&bind_group_layout, &camera_bind_group_layout],
                        &[],
                    ))
                    .vertex(VertexShaderState {
                        entry_point: "vs_main",
                        buffers: &[ModelVertex::desc(), InstanceRaw::desc()],
                    })
                    .fragment(FragmentShaderState {
                        entry_point: "frag_main",
                        targets: &[Some(ColorTargetState {
                            format: state.format(),
                            blend: Some(BlendState::REPLACE),
                            write_mask: ColorWrites::ALL,
                        })],
                    })
                    .primitive(PrimitiveState {
                        topology: PrimitiveTopology::TriangleList,
                        strip_index_format: None,
                        front_face: FrontFace::Ccw,
                        cull_mode: Some(Face::Back),
                        // Setting this to anything other than Fill requires Features::NON_FILL_POLYGON_MODE
                        polygon_mode: PolygonMode::Fill,
                        // Requires Features::DEPTH_CLIP_CONTROL
                        unclipped_depth: false,
                        // Requires Features::CONSERVATIVE_RASTERIZATION
                        conservative: false,
                    })
                    .multisample(MultisampleState {
                        count: 1,
                        mask: !0,
                        alpha_to_coverage_enabled: false,
                    })
                    .depth_stencil(DepthStencilState {
                        format: DEPTH_FORMAT,
                        depth_write_enabled: true,
                        depth_compare: CompareFunction::Less,
                        stencil: Default::default(),
                        bias: Default::default(),
                    })
                    .shader_src(ShaderModuleSources::Single(ShaderSource::Wgsl(
                        include_str!("shader.wgsl").into(),
                    ))),
            ),
            state.create_pipeline(
                PipelineBuilder::new()
                    .layout(&state.create_pipeline_layout(
                        &[&bind_group_layout, &camera_bind_group_layout],
                        &[],
                    ))
                    .vertex(VertexShaderState {
                        entry_point: "vs_main",
                        buffers: &[ModelVertex::desc(), InstanceRaw::desc()],
                    })
                    .fragment(FragmentShaderState {
                        entry_point: "frag_main",
                        targets: &[Some(ColorTargetState {
                            format: state.format(),
                            blend: Some(BlendState::REPLACE),
                            write_mask: ColorWrites::ALL,
                        })],
                    })
                    .primitive(PrimitiveState {
                        topology: PrimitiveTopology::TriangleList,
                        strip_index_format: None,
                        front_face: FrontFace::Ccw,
                        cull_mode: Some(Face::Back),
                        // Setting this to anything other than Fill requires Features::NON_FILL_POLYGON_MODE
                        polygon_mode: PolygonMode::Fill,
                        // Requires Features::DEPTH_CLIP_CONTROL
                        unclipped_depth: false,
                        // Requires Features::CONSERVATIVE_RASTERIZATION
                        conservative: false,
                    })
                    .multisample(MultisampleState {
                        count: 1,
                        mask: !0,
                        alpha_to_coverage_enabled: false,
                    })
                    .depth_stencil(DepthStencilState {
                        format: DEPTH_FORMAT,
                        depth_write_enabled: true,
                        depth_compare: CompareFunction::Less,
                        stencil: Default::default(),
                        bias: Default::default(),
                    })
                    .shader_src(ShaderModuleSources::Single(ShaderSource::Wgsl(
                        include_str!("new_shader.wgsl").into(),
                    ))),
            ),
        ];

    let mut state = AppState {
        camera,
        state: Arc::new(state),
        camera_uniform,
        camera_controller,
        camera_buffer,
    };

    let tmp_state = state.state.clone();
    let tmp_instance_buffer = instance_buffer.clone();
    thread::spawn(move || {
        let state = tmp_state;
        let instance_buffer = tmp_instance_buffer;
        loop {
            sleep(Duration::new(1, 0));
            let rand = rand::thread_rng().gen_range(0.5..1.0);
            const SPACE_BETWEEN: f32 = 3.0;
            let instances = (0..NUM_INSTANCES_PER_ROW)
                .flat_map(|z| {
                    (0..NUM_INSTANCES_PER_ROW).map(move |x| {
                        let x =
                            SPACE_BETWEEN * (x as f32 - NUM_INSTANCES_PER_ROW as f32 / 2.0) * rand;
                        let z =
                            SPACE_BETWEEN * (z as f32 - NUM_INSTANCES_PER_ROW as f32 / 2.0) * rand;

                        let position = Vector3 { x, y: 0.0, z };

                        let rotation = if position.is_zero() {
                            Quaternion::from_axis_angle(Vector3::unit_z(), Deg(0.0))
                        } else {
                            Quaternion::from_axis_angle(position.normalize(), Deg(45.0))
                        };

                        Instance { position, rotation }
                    })
                })
                .collect::<Vec<_>>();

            let instance_data = instances.iter().map(Instance::to_raw).collect::<Vec<_>>();
            state.write_buffer(&instance_buffer, 0, &instance_data);
        }
    });

    event_loop.run(move |event, _, control_flow| match event {
        Event::NewEvents(_) => {}
        Event::WindowEvent {
            ref event,
            window_id,
        } if window_id == window.id() => {
            if !state.input(event) {
                match event {
                    WindowEvent::Resized(size) => {
                        if !state.state.resize(*size) {
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
                        if !state.state.resize(**new_inner_size) {
                            println!("Couldn't resize!");
                        }
                    }
                    WindowEvent::ThemeChanged(_) => {}
                    WindowEvent::Ime(_) => {}
                    WindowEvent::Occluded(_) => {}
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
            let depth_tex = state.state.create_depth_texture(DEPTH_FORMAT);
            let depth_view = depth_tex.create_view(&TextureViewDescriptor::default());
            let bind_group = bind_group.clone();
            let bind_group = bind_group.lock();
            match state.state.render(|view, mut encoder, state| {
                {
                    let attachments = [Some(RenderPassColorAttachment {
                        view: &view,
                        resolve_target: None,
                        ops: Operations {
                            load: LoadOp::Clear(Color::GREEN),
                            store: true,
                        },
                    })];
                    let mut render_pass = state.create_render_pass(
                        &mut encoder,
                        &attachments,
                        Some(RenderPassDepthStencilAttachment {
                            view: &depth_view,
                            depth_ops: Some(Operations {
                                load: LoadOp::Clear(1.0),
                                store: true,
                            }),
                            stencil_ops: None,
                        }),
                    );
                    if dynamic_color {
                        render_pass.set_pipeline(&pipelines[1]);
                    } else {
                        render_pass.set_pipeline(&pipelines[0]);
                    }

                    render_pass.set_bind_group(0, &bind_group, &[]);
                    render_pass.set_bind_group(1, &camera_bind_group, &[]);
                    render_pass.set_vertex_buffer(1, instance_buffer.slice(..));

                    render_pass
                        .draw_mesh_instanced(&obj_model.meshes[0], 0..instances.len() as u32);
                }
                encoder
            }) {
                Ok(_) => {}
                // Reconfigure the surface if lost
                Err(SurfaceError::Lost) => {
                    let size = state.state.size();
                    state.state.resize(size);
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

struct Camera {
    eye: Point3<f32>,
    target: Point3<f32>,
    up: Vector3<f32>,
    aspect: f32,
    fovy: f32,
    znear: f32,
    zfar: f32,
}

impl Camera {
    fn build_view_projection_matrix(&self) -> Matrix4<f32> {
        let view = Matrix4::look_at_rh(self.eye, self.target, self.up);
        let proj = perspective(Deg(self.fovy), self.aspect, self.znear, self.zfar);

        return OPENGL_TO_WGPU_MATRIX * proj * view;
    }
}

#[rustfmt::skip]
pub const OPENGL_TO_WGPU_MATRIX: Matrix4<f32> = Matrix4::new(
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.0,
    0.0, 0.0, 0.5, 1.0,
);

// We need this for Rust to store our data correctly for the shaders
#[repr(C)]
// This is so we can store this in a buffer
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform {
    // We can't use cgmath with bytemuck directly so we'll have
    // to convert the Matrix4 into a 4x4 f32 array
    view_proj: [[f32; 4]; 4],
}

impl CameraUniform {
    fn new() -> Self {
        Self {
            view_proj: Matrix4::identity().into(),
        }
    }

    fn update_view_proj(&mut self, camera: &Camera) {
        self.view_proj = camera.build_view_projection_matrix().into();
    }
}

struct CameraController {
    speed: f32,
    is_forward_pressed: bool,
    is_backward_pressed: bool,
    is_left_pressed: bool,
    is_right_pressed: bool,
}

impl CameraController {
    fn new(speed: f32) -> Self {
        Self {
            speed,
            is_forward_pressed: false,
            is_backward_pressed: false,
            is_left_pressed: false,
            is_right_pressed: false,
        }
    }

    fn process_events(&mut self, event: &WindowEvent) -> bool {
        match event {
            WindowEvent::KeyboardInput {
                input:
                    KeyboardInput {
                        state,
                        virtual_keycode: Some(keycode),
                        ..
                    },
                ..
            } => {
                let is_pressed = *state == ElementState::Pressed;
                match keycode {
                    VirtualKeyCode::W | VirtualKeyCode::Up => {
                        self.is_forward_pressed = is_pressed;
                        true
                    }
                    VirtualKeyCode::A | VirtualKeyCode::Left => {
                        self.is_left_pressed = is_pressed;
                        true
                    }
                    VirtualKeyCode::S | VirtualKeyCode::Down => {
                        self.is_backward_pressed = is_pressed;
                        true
                    }
                    VirtualKeyCode::D | VirtualKeyCode::Right => {
                        self.is_right_pressed = is_pressed;
                        true
                    }
                    _ => false,
                }
            }
            _ => false,
        }
    }

    fn update_camera(&self, camera: &mut Camera) {
        let forward = camera.target - camera.eye;
        let forward_norm = forward.normalize();
        let forward_mag = forward.magnitude();

        // Prevents glitching when camera gets too close to the
        // center of the scene.
        if self.is_forward_pressed && forward_mag > self.speed {
            camera.eye += forward_norm * self.speed;
        }
        if self.is_backward_pressed {
            camera.eye -= forward_norm * self.speed;
        }

        let right = forward_norm.cross(camera.up);

        // Redo radius calc in case the forward/backward is pressed.
        let forward = camera.target - camera.eye;
        let forward_mag = forward.magnitude();

        if self.is_right_pressed {
            // Rescale the distance between the target and eye so
            // that it doesn't change. The eye therefore still
            // lies on the circle made by the target and eye.
            camera.eye = camera.target - (forward + right * self.speed).normalize() * forward_mag;
        }
        if self.is_left_pressed {
            camera.eye = camera.target - (forward - right * self.speed).normalize() * forward_mag;
        }
    }
}

struct AppState {
    state: Arc<State>,
    camera: Camera,
    camera_buffer: Buffer,
    camera_uniform: CameraUniform,
    camera_controller: CameraController,
}

impl AppState {
    /// Returns whether the event has been fully processed or not
    pub fn input(&mut self, event: &WindowEvent) -> bool {
        self.camera_controller.process_events(event)
    }

    pub fn update(&mut self) {
        self.camera_controller.update_camera(&mut self.camera);
        self.camera_uniform.update_view_proj(&self.camera);
        self.state
            .write_buffer(&self.camera_buffer, 0, &[self.camera_uniform]);
    }
}

struct Instance {
    position: Vector3<f32>,
    rotation: Quaternion<f32>,
}

impl Instance {
    fn to_raw(&self) -> InstanceRaw {
        InstanceRaw {
            model: (Matrix4::from_translation(self.position) * Matrix4::from(self.rotation)).into(),
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct InstanceRaw {
    model: [[f32; 4]; 4],
}

impl InstanceRaw {
    fn desc<'a>() -> VertexBufferLayout<'a> {
        VertexBufferLayout {
            array_stride: size_of::<InstanceRaw>() as BufferAddress,
            // We need to switch from using a step mode of Vertex to Instance
            // This means that our shaders will only change to use the next
            // instance when the shader starts processing a new instance
            step_mode: VertexStepMode::Instance,
            attributes: &[
                VertexAttribute {
                    offset: 0,
                    // While our vertex shader only uses locations 0, and 1 now, in later tutorials we'll
                    // be using 2, 3, and 4, for Vertex. We'll start at slot 5 not conflict with them later
                    shader_location: 5,
                    format: VertexFormat::Float32x4,
                },
                // A mat4 takes up 4 vertex slots as it is technically 4 vec4s. We need to define a slot
                // for each vec4. We'll have to reassemble the mat4 in
                // the shader.
                VertexAttribute {
                    offset: size_of::<[f32; 4]>() as BufferAddress,
                    shader_location: 6,
                    format: VertexFormat::Float32x4,
                },
                VertexAttribute {
                    offset: size_of::<[f32; 8]>() as BufferAddress,
                    shader_location: 7,
                    format: VertexFormat::Float32x4,
                },
                VertexAttribute {
                    offset: size_of::<[f32; 12]>() as BufferAddress,
                    shader_location: 8,
                    format: VertexFormat::Float32x4,
                },
            ],
        }
    }
}

const NUM_INSTANCES_PER_ROW: u32 = 10;
