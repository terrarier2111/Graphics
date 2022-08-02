use bytemuck::Pod;
use std::iter::once;
use std::num::NonZeroU32;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    Backends, BindGroupLayout, Buffer, BufferUsages, ColorTargetState, CommandEncoderDescriptor,
    DepthStencilState, Device, DeviceDescriptor, FragmentState, Instance, MultisampleState,
    PipelineLayoutDescriptor, PowerPreference, PresentMode, PrimitiveState, PushConstantRange,
    Queue, RenderPass, RenderPassColorAttachment, RenderPassDepthStencilAttachment,
    RenderPassDescriptor, RenderPipeline, RenderPipelineDescriptor, RequestAdapterOptions,
    RequestDeviceError, ShaderModule, ShaderModuleDescriptor, ShaderSource, Surface,
    SurfaceConfiguration, SurfaceError, TextureFormat, TextureUsages, TextureView,
    TextureViewDescriptor, VertexBufferLayout, VertexState,
};
use winit::dpi::PhysicalSize;
use winit::event::WindowEvent;
use winit::window::Window;

pub struct State {
    surface: Surface,
    device: Device,
    queue: Queue,
    config: SurfaceConfiguration,
    render_pipelines: Box<[RenderPipeline]>, // FIXME: should this be an Arc?
}

impl State {
    pub async fn new(
        window: &Window,
        power_pref: PowerPreference,
        descriptor: &DeviceDescriptor<'_>,
        present_mode: PresentMode,
        texture_usages: TextureUsages,
    ) -> Result<Option<Self>, RequestDeviceError> {
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
            }));
        }
        Ok(None)
    }

    pub fn setup_pipelines(&mut self, pipelines: Vec<PipelineState<'_>>) {
        let mut finished_pipelines = vec![];
        for pipeline in pipelines {
            let render_pipeline_layout =
                self.device
                    .create_pipeline_layout(&PipelineLayoutDescriptor {
                        label: None,
                        bind_group_layouts: pipeline.bind_group_layouts,
                        push_constant_ranges: pipeline.push_constant_ranges,
                    });
            let shaders = pipeline.shader_sources.to_modules(&self.device);

            let render_pipeline = self
                .device
                .create_render_pipeline(&RenderPipelineDescriptor {
                    label: None,
                    layout: Some(&render_pipeline_layout),
                    vertex: VertexState {
                        module: shaders.vertex_module(),
                        entry_point: pipeline.vertex_shader.entry_point,
                        buffers: pipeline.vertex_shader.buffers,
                    },
                    fragment: pipeline
                        .fragment_shader
                        .map(|fragment_shader| FragmentState {
                            module: shaders.fragment_module(),
                            entry_point: fragment_shader.entry_point,
                            targets: fragment_shader.targets,
                        }),
                    primitive: pipeline.primitive,
                    depth_stencil: pipeline.depth_stencil,
                    multisample: pipeline.multisample,
                    multiview: pipeline.multiview,
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

    pub fn render<'a, F: FnOnce(&TextureView) -> Box<[Option<RenderPassColorAttachment>]>>(
        &'a self,
        handler: impl RenderPassHandler<'a>,
        f: F,
        depth_stencil_attachment: Option<RenderPassDepthStencilAttachment>,
    ) -> Result<(), SurfaceError> {
        let output = self.surface.get_current_texture()?;
        // get a view of the current texture in order to render on it
        let view = output
            .texture
            .create_view(&mut TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor::default());
        // create a render pass in the encoder
        let color_attachments = f(&view);
        let render_pass = encoder.begin_render_pass(&RenderPassDescriptor {
            label: None,
            color_attachments: &color_attachments,
            depth_stencil_attachment,
        });
        handler.handle(render_pass, &self.render_pipelines);
        // FIXME: allow for more usage of command encoder

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
            }
            ShaderModuleSources::Multi(vertex_src, fragment_src) => ShaderModules::Multi(
                device.create_shader_module(ShaderModuleDescriptor {
                    label: None,
                    source: vertex_src,
                }),
                device.create_shader_module(ShaderModuleDescriptor {
                    label: None,
                    source: fragment_src,
                }),
            ),
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
    fn handle<'b: 'c, 'c>(
        self,
        render_pass: RenderPass<'c>,
        render_pipelines: &'b Box<[RenderPipeline]>,
    ) where
        'a: 'b;
}
