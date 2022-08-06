use crate::resources::{load_string, load_texture};
use crate::{State, TextureAspect};
use anyhow::Result;
use image::{DynamicImage, GenericImageView};
use std::io::{BufReader, Cursor};
use std::mem::size_of;
use std::ops::Range;
use tobj::LoadOptions;
use wgpu::{AddressMode, BindGroup, BindGroupEntry, BindGroupLayout, BindingResource, Buffer, BufferAddress, BufferUsages, FilterMode, IndexFormat, RenderPass, Sampler, SamplerDescriptor, Texture, TextureDimension, TextureFormat, TextureUsages, TextureView, TextureViewDescriptor, VertexAttribute, VertexBufferLayout, VertexFormat, VertexStepMode};

pub trait Vertex {
    fn desc<'a>() -> VertexBufferLayout<'a>;
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ModelVertex {
    pub position: [f32; 3],
    pub tex_coords: [f32; 2],
    pub normal: [f32; 3],
}

impl Vertex for ModelVertex {
    fn desc<'a>() -> VertexBufferLayout<'a> {
        VertexBufferLayout {
            array_stride: size_of::<ModelVertex>() as BufferAddress,
            step_mode: VertexStepMode::Vertex,
            attributes: &[
                VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: VertexFormat::Float32x3,
                },
                VertexAttribute {
                    offset: size_of::<[f32; 3]>() as BufferAddress,
                    shader_location: 1,
                    format: VertexFormat::Float32x2,
                },
                VertexAttribute {
                    offset: size_of::<[f32; 5]>() as BufferAddress,
                    shader_location: 2,
                    format: VertexFormat::Float32x3,
                },
            ],
        }
    }
}

pub struct Model {
    pub meshes: Vec<Mesh>,
    pub materials: Vec<Material>,
}

impl Model {
    pub async fn load_from(
        file_name: &str,
        state: &State,
        layout: &BindGroupLayout,
    ) -> Result<Self> {
        let obj_text = load_string(file_name).await?;
        let obj_cursor = Cursor::new(obj_text);
        let mut obj_reader = BufReader::new(obj_cursor);

        let (models, obj_materials) = tobj::load_obj_buf_async(
            &mut obj_reader,
            &LoadOptions {
                triangulate: true,
                single_index: true,
                ..Default::default()
            },
            |p| async move {
                let mat_text = load_string(&p).await.unwrap();
                tobj::load_mtl_buf(&mut BufReader::new(Cursor::new(mat_text)))
            },
        )
        .await?;

        let mut materials = Vec::new();
        for m in obj_materials? {
            let diffuse_texture = load_texture(&m.diffuse_texture, state).await?;
            let bind_group = state.create_bind_group(
                layout,
                &[
                    BindGroupEntry {
                        binding: 0,
                        resource: BindingResource::TextureView(&diffuse_texture.view),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: BindingResource::Sampler(&diffuse_texture.sampler),
                    },
                ],
            );

            materials.push(Material {
                name: m.name,
                diffuse_texture,
                bind_group,
            })
        }

        let meshes = models
            .into_iter()
            .map(|m| {
                let vertices = (0..m.mesh.positions.len() / 3)
                    .map(|i| ModelVertex {
                        position: [
                            m.mesh.positions[i * 3],
                            m.mesh.positions[i * 3 + 1],
                            m.mesh.positions[i * 3 + 2],
                        ],
                        tex_coords: [m.mesh.texcoords[i * 2], m.mesh.texcoords[i * 2 + 1]],
                        normal: [
                            m.mesh.normals[i * 3],
                            m.mesh.normals[i * 3 + 1],
                            m.mesh.normals[i * 3 + 2],
                        ],
                    })
                    .collect::<Vec<_>>();

                let vertex_buffer = state.create_buffer(&vertices, BufferUsages::VERTEX);
                let index_buffer = state.create_buffer(&m.mesh.indices, BufferUsages::INDEX);

                Mesh {
                    name: file_name.to_string(),
                    vertex_buffer,
                    index_buffer,
                    num_elements: m.mesh.indices.len() as u32,
                    material: m.mesh.material_id.unwrap_or(0),
                }
            })
            .collect::<Vec<_>>();

        Ok(Self { meshes, materials })
    }
}

pub struct Material {
    pub name: String,
    pub diffuse_texture: ContainedTexture,
    pub bind_group: BindGroup,
}

pub struct Mesh {
    pub name: String,
    pub vertex_buffer: Buffer,
    pub index_buffer: Buffer,
    pub num_elements: u32,
    pub material: usize,
}

pub struct ContainedTexture {
    pub texture: Texture,
    pub view: TextureView,
    pub sampler: Sampler,
}

impl ContainedTexture {
    pub fn from_bytes(state: &State, bytes: &[u8]) -> Result<Self> {
        let img = image::load_from_memory(bytes)?;
        Ok(Self::from_image(state, &img))
    }

    pub fn from_image(state: &State, img: &DynamicImage) -> Self {
        let rgba = img.to_rgba8();
        let dimensions = img.dimensions();
        let tex = state.create_texture(
            &rgba,
            dimensions,
            TextureFormat::Rgba8UnormSrgb,
            TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            TextureDimension::D2,
            TextureAspect::All,
            None,
            None,
        );

        let view = tex.create_view(&TextureViewDescriptor::default());
        let sampler = state.device().create_sampler(&SamplerDescriptor {
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Nearest,
            mipmap_filter: FilterMode::Nearest,
            ..Default::default()
        });

        Self {
            texture: tex,
            view,
            sampler,
        }
    }
}

pub trait DrawModel<'a> {
    fn draw_mesh(&mut self, mesh: &'a Mesh);
    fn draw_mesh_instanced(
        &mut self,
        mesh: &'a Mesh,
        instances: Range<u32>,
    );
}
impl<'a, 'b: 'a> DrawModel<'b> for RenderPass<'a> {
    fn draw_mesh(&mut self, mesh: &'b Mesh) {
        self.draw_mesh_instanced(mesh, 0..1);
    }

    fn draw_mesh_instanced(
        &mut self,
        mesh: &'b Mesh,
        instances: Range<u32>,
    ){
        self.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
        self.set_index_buffer(mesh.index_buffer.slice(..), IndexFormat::Uint32);
        self.draw_indexed(0..mesh.num_elements, 0, instances);
    }
}

// FIXME: we could generalize this by using a trait Drawable and DrawableIndexed which provide us with methods to get the buffers we need
// FIXME: and implementing a Draw and DrawIndexed trait for RenderPass which allows it to draw all types of Drawable and DrawableIndexed
