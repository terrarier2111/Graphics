use bytemuck::Pod;
use parking_lot::RwLock;
use std::iter::once;
use std::num::NonZeroU32;
use std::sync::atomic::{AtomicBool, Ordering};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    Backends, BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout,
    BindGroupLayoutDescriptor, BindGroupLayoutEntry, Buffer, BufferAddress, BufferUsages,
    ColorTargetState, CommandEncoderDescriptor, DepthStencilState, Device, DeviceDescriptor,
    Extent3d, Features, FragmentState, ImageCopyTexture, ImageDataLayout, Instance, Limits,
    MultisampleState, Origin3d, PipelineLayoutDescriptor, PowerPreference, PresentMode,
    PrimitiveState, PushConstantRange, Queue, RenderPass, RenderPassColorAttachment,
    RenderPassDepthStencilAttachment, RenderPassDescriptor, RenderPipeline,
    RenderPipelineDescriptor, RequestAdapterOptions, RequestDeviceError, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, Surface, SurfaceConfiguration, SurfaceError, Texture,
    TextureAspect, TextureDescriptor, TextureDimension, TextureFormat, TextureUsages, TextureView,
    TextureViewDescriptor, VertexAttribute, VertexBufferLayout, VertexFormat, VertexState,
};
use winit::dpi::PhysicalSize;
use winit::window::Window;

pub struct State {
    surface: Surface,
    device: Device,
    queue: Queue,
    config: RwLock<SurfaceConfiguration>,
    render_pipelines: Box<[RenderPipeline]>, // FIXME: should this be an Arc?
    surface_texture_alive: AtomicBool,
}

impl State {
    pub async fn new(builder: StateBuilder<'_>) -> Result<Option<Self>, RequestDeviceError> {
        // FIXME: check if we can somehow choose a better/more descriptive return type
        let window = builder
            .window
            .expect("window has to be specified before building the state");
        let size = window.inner_size();
        // The instance is a handle to our GPU
        let instance = Instance::new(builder.backends); // used to create adapters and surfaces
        let surface = unsafe { instance.create_surface(window) };
        let adapter = instance // adapter is a handle to our graphics card
            .request_adapter(&RequestAdapterOptions {
                power_preference: builder.power_pref,
                force_fallback_adapter: false,
                compatible_surface: Some(&surface),
            })
            .await;
        if let Some(adapter) = adapter {
            let (device, queue) = adapter
                .request_device(
                    &DeviceDescriptor {
                        label: None,
                        features: builder.requirements.features,
                        limits: builder.requirements.limits,
                    },
                    None,
                )
                .await?;

            let pref_format = surface.get_supported_formats(&adapter)[0];
            let config = SurfaceConfiguration {
                usage: TextureUsages::RENDER_ATTACHMENT,
                format: pref_format,
                width: size.width,
                height: size.height,
                present_mode: builder.present_mode,
            };
            surface.configure(&device, &config);

            return Ok(Some(Self {
                surface,
                device,
                queue,
                config: RwLock::new(config),
                render_pipelines: Box::new([]),
                surface_texture_alive: Default::default(),
            }));
        }
        Ok(None)
    }

    pub fn setup_pipelines(&mut self, pipelines: Box<[PipelineState<'_>]>) {
        let mut finished_pipelines = Box::new_uninit_slice(pipelines.len());
        let tmp = Vec::from(pipelines);
        for (idx, pipeline) in tmp.into_iter().enumerate() {
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

            finished_pipelines[idx].write(render_pipeline);
        }
        // SAFETY: we can assume finished_pipelines is entirely init as
        // we just iterated over all its entries and populated them
        self.render_pipelines = unsafe { finished_pipelines.assume_init() };
    }

    /// Returns whether the resizing succeeded or not
    pub fn resize(&self, size: PhysicalSize<u32>) -> bool {
        if size.height > 0 && size.height > 0 {
            let mut config = self.config.write();
            config.width = size.width;
            config.height = size.height;
            // FIXME: should we verify that there exist no old textures?
            self.surface.configure(&self.device, &*config);
            true
        } else {
            false
        }
    }

    pub fn render<'a, F: FnOnce(&TextureView) -> Box<[Option<RenderPassColorAttachment>]>>(
        &'a self,
        handler: impl RenderPassHandler<'a>,
        color_provider: F,
        depth_stencil_attachment: Option<RenderPassDepthStencilAttachment>,
    ) -> Result<(), SurfaceError> {
        self.surface_texture_alive.store(true, Ordering::Release);
        let output = self.surface.get_current_texture()?;
        // get a view of the current texture in order to render on it
        let view = output
            .texture
            .create_view(&mut TextureViewDescriptor::default()); // FIXME: do we need a way to parameterize this?
        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor::default());
        // create a render pass in the encoder
        let color_attachments = color_provider(&view);
        let render_pass = encoder.begin_render_pass(&RenderPassDescriptor {
            label: None,
            color_attachments: &color_attachments,
            depth_stencil_attachment,
        });
        handler.handle(render_pass, &self.render_pipelines);
        // FIXME: allow for more usage of command encoder

        self.queue.submit(once(encoder.finish()));
        output.present();
        self.surface_texture_alive.store(false, Ordering::Release);

        Ok(())
    }

    pub fn size(&self) -> PhysicalSize<u32> {
        let config = self.config.read();
        PhysicalSize::new(config.width, config.height)
    }

    pub fn format(&self) -> TextureFormat {
        let config = self.config.read();
        config.format.clone()
    }

    pub fn try_update_present_mode(&self, present_mode: PresentMode) -> bool {
        if !self.surface_texture_alive.load(Ordering::Acquire) {
            let mut config = self.config.write();
            config.present_mode = present_mode;
            self.surface.configure(&self.device, &*config);
            true
        } else {
            false
        }
    }

    pub fn update_present_mode(&self, present_mode: PresentMode) {
        while !self.try_update_present_mode(present_mode) {
            // do nothing, as we just want to update the present mode
        }
    }

    #[inline]
    pub const fn device(&self) -> &Device {
        &self.device
    }

    #[inline]
    pub const fn queue(&self) -> &Queue {
        &self.queue
    }

    #[inline]
    pub const fn surface(&self) -> &Surface {
        &self.surface
    }

    pub fn dimensions(&self) -> (u32, u32) {
        let config = self.config.read();
        (config.width, config.height)
    }

    pub fn create_buffer<T: Pod>(&self, content: &[T], usage: BufferUsages) -> Buffer {
        // FIXME: should we switch from Pod to NoUninit?
        self.device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(content),
            usage,
        })
    }

    pub fn create_texture(&self, builder: TextureBuilder) -> Texture {
        let mip_info = builder.mip_info;
        let dimensions = builder
            .dimensions
            .expect("dimensions have to be specified before building the texture");
        let texture_size = Extent3d {
            width: dimensions.0,
            height: dimensions.1,
            depth_or_array_layers: builder.depth_or_array_layers,
        };
        let format = builder
            .format
            .expect("format has to be specified before building the texture");
        let diffuse_texture = self.device.create_texture(&TextureDescriptor {
            // All textures are stored as 3D, we represent our 2D texture
            // by setting depth to 1.
            size: texture_size,
            mip_level_count: mip_info.mip_level_count, // We'll talk about this a little later
            sample_count: builder.sample_count,
            dimension: builder
                .texture_dimension
                .expect("texture dimension has to be specified before building the texture"),
            // Most images are stored using sRGB so we need to reflect that here.
            format,
            // COPY_DST means that we want to copy data to this texture
            usage: builder.usages,
            label: None,
        });
        self.queue.write_texture(
            // Tells wgpu where to copy the pixel data
            ImageCopyTexture {
                texture: &diffuse_texture,
                mip_level: mip_info.target_mip_level,
                origin: mip_info.origin,
                aspect: builder.aspect,
            },
            // The actual pixel data
            builder
                .data
                .expect("data has to be specified before building the texture"),
            // The layout of the texture
            ImageDataLayout {
                offset: 0,
                bytes_per_row: NonZeroU32::new(format.describe().block_size as u32 * dimensions.0),
                rows_per_image: NonZeroU32::new(dimensions.1),
            },
            texture_size,
        );

        diffuse_texture
    }

    pub fn create_bind_group_layout(&self, entries: &[BindGroupLayoutEntry]) -> BindGroupLayout {
        self.device
            .create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: None,
                entries,
            })
    }

    pub fn create_bind_group(
        &self,
        layout: &BindGroupLayout,
        entries: &[BindGroupEntry],
    ) -> BindGroup {
        self.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout,
            entries,
        })
    }

    pub fn write_buffer<T: Pod>(&self, buffer: &Buffer, offset: BufferAddress, data: &[T]) {
        self.queue
            .write_buffer(buffer, offset, bytemuck::cast_slice(data));
    }

    pub fn create_depth_texture(&self, format: TextureFormat) -> Texture {
        let (width, height) = self.dimensions();
        let size = Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        let texture_desc = TextureDescriptor {
            label: None,
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::RENDER_ATTACHMENT,
        };
        self.device.create_texture(&texture_desc)
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

#[derive(Default)]
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
    #[inline]
    pub fn new() -> Self {
        Self::default()
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

pub struct TextureBuilder<'a> {
    data: Option<&'a [u8]>,
    dimensions: Option<(u32, u32)>,
    format: Option<TextureFormat>,
    texture_dimension: Option<TextureDimension>,
    usages: TextureUsages,      // MAYBE(currently): we have a default
    aspect: TextureAspect,      // we have a default
    sample_count: u32,          // we have a default
    mip_info: MipInfo,          // we have a default
    depth_or_array_layers: u32, // we have a default
}

impl<'a> Default for TextureBuilder<'a> {
    fn default() -> Self {
        Self {
            data: None,
            dimensions: None,
            format: None,
            texture_dimension: None,
            usages: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            aspect: TextureAspect::All,
            sample_count: 1,
            mip_info: MipInfo::default(),
            depth_or_array_layers: 1,
        }
    }
}

impl<'a> TextureBuilder<'a> {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn data(mut self, data: &'a [u8]) -> Self {
        self.data = Some(data);
        self
    }

    pub fn dimensions(mut self, dimensions: (u32, u32)) -> Self {
        self.dimensions = Some(dimensions);
        self
    }

    pub fn format(mut self, format: TextureFormat) -> Self {
        self.format = Some(format);
        self
    }

    pub fn texture_dimension(mut self, texture_dimension: TextureDimension) -> Self {
        self.texture_dimension = Some(texture_dimension);
        self
    }

    /// The default value is TextureUsages::TEXTURE_BINDING
    /// NOTE: TextureUsages::COPY_DST gets appended to the usages
    pub fn usages(mut self, usages: TextureUsages) -> Self {
        self.usages = usages | TextureUsages::COPY_DST;
        self
    }

    /// The default value is `TextureAspect::All`
    pub fn aspect(mut self, aspect: TextureAspect) -> Self {
        self.aspect = aspect;
        self
    }

    /// The default value is 1
    pub fn sample_count(mut self, sample_count: u32) -> Self {
        self.sample_count = sample_count;
        self
    }

    /// The default value is `MipInfo::default()`
    pub fn mip_info(mut self, mip_info: MipInfo) -> Self {
        self.mip_info = mip_info;
        self
    }

    /// The default value is 1
    pub fn depth_or_array_layers(mut self, depth_or_array_layers: u32) -> Self {
        self.depth_or_array_layers = depth_or_array_layers;
        self
    }

    #[inline]
    pub fn build(self, state: &State) -> Texture {
        state.create_texture(self)
    }
}

pub struct StateBuilder<'a> {
    window: Option<&'a Window>,
    power_pref: PowerPreference,      // we have a default
    present_mode: PresentMode,        // we have a default
    requirements: DeviceRequirements, // we have a default
    backends: Backends,               // we have a default
}

impl<'a> Default for StateBuilder<'a> {
    fn default() -> Self {
        Self {
            backends: Backends::all(),
            window: None,
            power_pref: Default::default(),
            present_mode: Default::default(),
            requirements: Default::default(),
        }
    }
}

impl<'a> StateBuilder<'a> {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn window(mut self, window: &'a Window) -> Self {
        self.window = Some(window);
        self
    }

    /// The default value is `PowerPreference::LowPower`
    pub fn power_pref(mut self, power_pref: PowerPreference) -> Self {
        self.power_pref = power_pref;
        self
    }

    /// The default value is `PresentMode::Fifo`
    pub fn present_mode(mut self, present_mode: PresentMode) -> Self {
        self.present_mode = present_mode;
        self
    }

    // FIXME: should we rename this to `requirements`?
    /// The default value is `DeviceRequirements::default()`
    pub fn device_requirements(mut self, requirements: DeviceRequirements) -> Self {
        self.requirements = requirements;
        self
    }

    /// The default value is `Backends::all()`
    pub fn backends(mut self, backends: Backends) -> Self {
        self.backends = backends;
        self
    }

    #[inline]
    pub async fn build(self) -> Result<Option<State>, RequestDeviceError> {
        State::new(self).await
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

pub struct MipInfo {
    pub origin: Origin3d,
    pub target_mip_level: u32,
    pub mip_level_count: u32,
}

impl Default for MipInfo {
    fn default() -> Self {
        Self {
            origin: Origin3d::ZERO,
            target_mip_level: 0,
            mip_level_count: 1,
        }
    }
}

#[derive(Default)]
pub struct DeviceRequirements {
    pub features: Features,
    pub limits: Limits,
}

pub const fn matrix<const COLUMNS: usize>(
    offset: u64,
    location: u32,
    format: VertexFormat,
) -> [VertexAttribute; COLUMNS] {
    let mut ret = [VertexAttribute {
        format,
        offset: 0,
        shader_location: 0,
    }; COLUMNS];

    let mut x = 0;
    while COLUMNS > x {
        ret[x] = VertexAttribute {
            format,
            offset: (offset + format.size() * x as u64) as BufferAddress,
            shader_location: location + x as u32,
        };
        x += 1;
    }

    ret
}
