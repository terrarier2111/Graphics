use std::iter::once;
use std::marker::PhantomData;
use std::num::NonZeroU32;
use std::rc::Rc;
use bytemuck::Pod;
use wgpu::{Backends, BindGroupLayout, BlendState, Buffer, BufferUsages, Color, ColorTargetState, ColorWrites, CommandEncoder, CommandEncoderDescriptor, DepthStencilState, Device, DeviceDescriptor, FragmentState, Instance, LoadOp, MultisampleState, Operations, PipelineLayoutDescriptor, PowerPreference, PresentMode, PrimitiveState, PushConstantRange, Queue, RenderPass, RenderPassColorAttachment, RenderPassDescriptor, RenderPipeline, RenderPipelineDescriptor, RequestAdapterOptions, RequestDeviceError, ShaderModule, ShaderModuleDescriptor, ShaderSource, Surface, SurfaceConfiguration, SurfaceError, TextureFormat, TextureUsages, TextureView, TextureViewDescriptor, VertexBufferLayout, VertexState};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use winit::dpi::PhysicalSize;
use winit::event::WindowEvent;
use winit::window::Window;

pub struct State/*<'a>*/ {
    surface: Surface,
    device: Device,
    queue: Queue,
    config: SurfaceConfiguration,
    render_pipelines: Box<[RenderPipeline]>, // FIXME: should this be an Arc?
    // _phantom_data: PhantomData<&'a ()>,
}

impl/*<'a>*/ State/*<'a>*/ {
    pub async fn new(
        window: &Window,
        power_pref: PowerPreference,
        descriptor: &DeviceDescriptor<'_>,
        present_mode: PresentMode,
        texture_usages: TextureUsages,
    ) -> Result<Option<Self/*State<'a>*/>, RequestDeviceError> {
        // FIXME: check if we can somehow choose a better/more descriptive return type
        let size = window.inner_size();
        // The instance is a handle to our GPU
        // Backends::all => Vulkan + Metal + DX12 + Browser WebGPU
        let instance = Instance::new(Backends::all()); // used to create adapters and surfaces
        let surface = unsafe { instance.create_surface(window) };
        let adapter = instance // adapter is a handle to our graphics card
            .request_adapter(&RequestAdapterOptions {
                power_preference: power_pref,
                force_fallback_adapter: false,
                compatible_surface: Some(&surface),
            })
            .await;
        if let Some(adapter) = adapter {
            let (device, queue) = adapter.request_device(descriptor, None).await?;

            let pref_format = surface.get_supported_formats(&adapter)[0];
            let config = SurfaceConfiguration {
                usage: texture_usages,
                format: pref_format,
                width: size.width,
                height: size.height,
                present_mode,
            };
            surface.configure(&device, &config);

            return Ok(Some(Self {
                surface,
                device,
                queue,
                config,
                render_pipelines: Box::new([]),
                // _phantom_data: Default::default(),
            }));
        }
        Ok(None)
    }

    pub fn setup_pipelines(&mut self, pipelines: Vec<PipelineState<'_>>) {
        let mut finished_pipelines = vec![];
        for pipeline in pipelines {
            let render_pipeline_layout =
                self.device.create_pipeline_layout(&PipelineLayoutDescriptor {
                    label: None,
                    bind_group_layouts: pipeline.bind_group_layouts,
                    push_constant_ranges: pipeline.push_constant_ranges,
                });
            let shaders = pipeline.shader_sources.to_modules(&self.device);

            let render_pipeline = self.device.create_render_pipeline(&RenderPipelineDescriptor {
                label: None,
                layout: Some(&render_pipeline_layout),
                vertex: VertexState {
                    module: shaders.vertex_module(),
                    entry_point: pipeline.vertex_shader.entry_point,
                    buffers: pipeline.vertex_shader.buffers,
                },
                fragment: pipeline.fragment_shader.map(|fragment_shader| {
                    FragmentState {
                        module: shaders.fragment_module(),
                        entry_point: fragment_shader.entry_point,
                        targets: fragment_shader.targets,
                    }
                }),
                primitive: pipeline.primitive,
                depth_stencil: pipeline.depth_stencil, // 1.
                multisample: pipeline.multisample,
                multiview: pipeline.multiview, // 5.
            });

            finished_pipelines.push(render_pipeline);
        }
        self.render_pipelines = finished_pipelines.into_boxed_slice();
    }

    /// Returns whether the resizing succeeded or not
    pub fn resize(&mut self, size: PhysicalSize<u32>) -> bool {
        if size.height > 0 && size.height > 0 {
            self.config.width = size.width;
            self.config.height = size.height;
            self.surface.configure(&self.device, &self.config);
            true
        } else {
            false
        }
    }

    /// Returns whether the event has been fully processed or not
    pub fn input(&mut self, event: &WindowEvent) -> bool {
        false
    }

    pub fn update(&mut self) {}

    /*
    pub fn render<'a: 'b, 'b: 'c, 'c: 'd, 'd, /*'b, *//*'a, */F: /*FnMut*/Fn/*Once*/(&/*'a*//*'b */mut RenderPass<'c>, &'b Box<[RenderPipeline]>/*<'b>*/) + 'd/* -> Box<[Rc<RenderPipeline>]> + 'a*//* + 'a*/>(&'a self, f: F) -> Result<(), SurfaceError> {
        let output = self.surface.get_current_texture()?;
        // get a view of the current texture in order to render on it
        let view = output
            .texture
            .create_view(&mut TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor::default());
        /*{
            // create a render pass in the encoder
            let mut render_pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(RenderPassColorAttachment { // FIXME: parameterize this!
                    view: &view,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Clear(Color::RED),
                        store: true,
                    },
                })],
                depth_stencil_attachment: None,
            });

            f(&mut render_pass, &self.render_pipelines);
        }*/
        let borrowed_encoder = &mut encoder;
        do_render_pass(&self, borrowed_encoder, &view, f);
        fn do_render_pass<'a: 'b, 'b: 'c, 'c: 'd, 'd: 'e, 'e: 'f, 'f, F: /*FnMut*/Fn/*Once*/(&'e mut RenderPass<'d>, &'b Box<[RenderPipeline]>) + 'f>(state: &'a State, encoder: &'c mut CommandEncoder, view: &'c TextureView, f: F) {
            // create a render pass in the encoder
            let mut render_pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(RenderPassColorAttachment { // FIXME: parameterize this!
                    view: &view,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Clear(Color::RED),
                        store: true,
                    },
                })],
                depth_stencil_attachment: None,
            });

            f(&mut render_pass, &state.render_pipelines);
            drop(render_pass);
        }

        self.queue.submit(once(encoder.finish()));
        output.present();

        Ok(())
    }*/

    pub fn render<'a>(&'a self, mut handler: impl RenderPassHandler<'a>) -> Result<(), SurfaceError> {
        let output = self.surface.get_current_texture()?;
        // get a view of the current texture in order to render on it
        let view = output
            .texture
            .create_view(&mut TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor::default());
        /*{
            // create a render pass in the encoder
            let mut render_pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(RenderPassColorAttachment { // FIXME: parameterize this!
                    view: &view,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Clear(Color::RED),
                        store: true,
                    },
                })],
                depth_stencil_attachment: None,
            });

            f(&mut render_pass, &self.render_pipelines);
        }*/
        // create a render pass in the encoder
        let render_pass = encoder.begin_render_pass(&RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(RenderPassColorAttachment { // FIXME: parameterize this!
                view: &view,
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::Clear(Color::BLACK),
                    store: true,
                },
            })],
            depth_stencil_attachment: None,
        });
        handler.handle(render_pass, &self.render_pipelines);

        self.queue.submit(once(encoder.finish()));
        output.present();

        Ok(())
    }

    #[inline]
    pub const fn size(&self) -> PhysicalSize<u32> {
        PhysicalSize::new(self.config.width, self.config.height)
    }

    #[inline]
    pub const fn format(&self) -> &TextureFormat {
        &self.config.format
    }

    pub fn create_buffer<T: Pod>(&self, content: &[T], usage: BufferUsages) -> Buffer {
        self.device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(content),
            usage,
        })
    }

    /*s
    #[inline]
    pub const fn render_pipeline(&self, index: usize) -> Option<Rc<RenderPipeline>> {
        self.render_pipelines.get(index).map(|pipeline| pipeline.clone())
    }*/

    /*
    pub fn try_set_render_pipeline/*<'a, 'b: 'a>*/(&/*'a *//*'a */self, render_pass: &/*'a */mut RenderPass/*<'a>*/, idx: usize) -> bool {
        if let Some(render_pipeline) = self.render_pipelines.get(idx) {
            render_pass.set_pipeline(render_pipeline);
            true
        } else {
            false
        }
    }*/

}

pub struct PipelineState<'a> {
    vertex_shader: VertexShaderState<'a>,
    fragment_shader: Option<FragmentShaderState<'a>>,
    primitive: PrimitiveState,
    bind_group_layouts: &'a [&'a BindGroupLayout],
    push_constant_ranges: &'a [PushConstantRange],
    depth_stencil: Option<DepthStencilState>,
    multisample: MultisampleState,
    multiview: Option<NonZeroU32>,
    shader_sources: ShaderModuleSources<'a>,
}

pub struct PipelineStateBuilder<'a> {
    vertex_shader: Option<VertexShaderState<'a>>,
    fragment_shader: Option<FragmentShaderState<'a>>,
    primitive: Option<PrimitiveState>,
    bind_group_layouts: Option<&'a [&'a BindGroupLayout]>,
    push_constant_ranges: Option<&'a [PushConstantRange]>,
    depth_stencil: Option<DepthStencilState>,
    multisample: Option<MultisampleState>,
    multiview: Option<NonZeroU32>,
    shader_sources: Option<ShaderModuleSources<'a>>,
}

impl<'a> PipelineStateBuilder<'a> {

    pub fn new() -> Self {
        Self {
            vertex_shader: None,
            fragment_shader: None,
            primitive: None,
            bind_group_layouts: None,
            push_constant_ranges: None,
            depth_stencil: None,
            multisample: None,
            multiview: None,
            shader_sources: None,
        }
    }

    pub fn vertex(mut self, vertex_shader: VertexShaderState<'a>) -> Self {
        self.vertex_shader = Some(vertex_shader);
        self
    }

    pub fn fragment(mut self, fragment_shader: FragmentShaderState<'a>) -> Self {
        self.fragment_shader = Some(fragment_shader);
        self
    }

    pub fn primitive(mut self, primitive: PrimitiveState) -> Self {
        self.primitive = Some(primitive);
        self
    }

    pub fn bind_group_layouts(mut self, bind_group_layouts: &'a [&'a BindGroupLayout]) -> Self {
        self.bind_group_layouts = Some(bind_group_layouts);
        self
    }

    pub fn push_constant_ranges(mut self, push_constant_ranges: &'a [PushConstantRange]) -> Self {
        self.push_constant_ranges = Some(push_constant_ranges);
        self
    }

    pub fn depth_stencil(mut self, depth_stencil: DepthStencilState) -> Self {
        self.depth_stencil = Some(depth_stencil);
        self
    }

    pub fn multisample(mut self, multisample: MultisampleState) -> Self {
        self.multisample = Some(multisample);
        self
    }

    pub fn multiview(mut self, multiview: NonZeroU32) -> Self {
        self.multiview = Some(multiview);
        self
    }

    pub fn shader_src(mut self, shader_sources: ShaderModuleSources<'a>) -> Self {
        self.shader_sources = Some(shader_sources);
        self
    }

    pub fn build(self) -> PipelineState<'a> {
        PipelineState {
            vertex_shader: self.vertex_shader.unwrap(),
            fragment_shader: self.fragment_shader,
            primitive: self.primitive.unwrap(),
            bind_group_layouts: self.bind_group_layouts.unwrap(),
            push_constant_ranges: self.push_constant_ranges.unwrap(),
            depth_stencil: self.depth_stencil,
            multisample: self.multisample.unwrap(),
            multiview: self.multiview,
            shader_sources: self.shader_sources.unwrap(),
        }
    }

}

pub struct VertexShaderState<'a> {
    /// The name of the entry point in the compiled shader. There must be a function that returns
    /// void with this name in the shader.
    pub entry_point: &'a str,
    /// The format of any vertex buffers used with this pipeline.
    pub buffers: &'a [VertexBufferLayout<'a>],
}

pub struct FragmentShaderState<'a> {
    /// The name of the entry point in the compiled shader. There must be a function that returns
    /// void with this name in the shader.
    pub entry_point: &'a str,
    /// The color state of the render targets.
    pub targets: &'a [Option<ColorTargetState>],
}

/*
enum ShaderModuleSupply {
    Vertex(ShaderModule),
    Single(ShaderModule),
    Multi(ShaderModule, ShaderModule),
}

impl From<(ShaderSource, Option<ShaderSource>, &Device)> for ShaderModuleSupply {
    fn from(sources: (ShaderSource, Option<ShaderSource>, &Device)) -> Self {
        if let Some(frag_src) = sources.1 {
            // FIXME: detect same sources and map them to a single shader module
            ShaderModuleSupply::Multi(sources.2.create_shader_module(ShaderModuleDescriptor {
                label: None,
                source: sources.0,
            }), sources.2.create_shader_module(ShaderModuleDescriptor {
                label: None,
                source: frag_src,
            }))
        } else {
            ShaderModuleSupply::Vertex(sources.2.create_shader_module(ShaderModuleDescriptor {
                label: None,
                source: sources.0,
            }))
        }
    }
}*/

pub enum ShaderModuleSources<'a> {
    Single(ShaderSource<'a>),
    Multi(ShaderSource<'a>, ShaderSource<'a>),
}

impl<'a> ShaderModuleSources<'a> {

    fn to_modules(self, device: &Device) -> ShaderModules {
        match self {
            ShaderModuleSources::Single(src) => {
                ShaderModules::Single(device.create_shader_module(ShaderModuleDescriptor {
                    label: None,
                    source: src,
                }))
            },
            ShaderModuleSources::Multi(vertex_src, fragment_src) => {
                ShaderModules::Multi(
                    device.create_shader_module(ShaderModuleDescriptor {
                        label: None,
                        source: vertex_src,
                    }),
                    device.create_shader_module(ShaderModuleDescriptor {
                        label: None,
                        source: fragment_src,
                    }))
            },
        }
    }

}

enum ShaderModules {
    Single(ShaderModule),
    Multi(ShaderModule, ShaderModule),
}

impl ShaderModules {

    fn vertex_module(&self) -> &ShaderModule {
        match self {
            ShaderModules::Single(module) => &module,
            ShaderModules::Multi(vertex_module, _) => &vertex_module,
        }
    }

    fn fragment_module(&self) -> &ShaderModule {
        match self {
            ShaderModules::Single(module) => &module,
            ShaderModules::Multi(_, fragment_module) => &fragment_module,
        }
    }

}

pub trait RenderPassHandler<'a> {

    fn handle<'b: 'c, 'c>(&mut self, render_pass: RenderPass<'b>, render_pipelines: &'a Box<[RenderPipeline]>) where 'a: 'b;

}
