use crate::model::ContainedTexture;
use crate::State;
use anyhow::Result;
use std::fs::{read, read_to_string};
use std::path::Path;

pub async fn load_string(file_name: &str) -> Result<String> {
    let path = Path::new(env!("OUT_DIR")).join("res").join(file_name);
    let txt = read_to_string(path)?;

    Ok(txt)
}

pub async fn load_binary(file_name: &str) -> Result<Vec<u8>> {
    let path = Path::new(env!("OUT_DIR")).join("res").join(file_name);
    let data = read(path)?;

    Ok(data)
}

pub async fn load_texture(file_name: &str, state: &State) -> Result<ContainedTexture> {
    let data = load_binary(file_name).await?;
    ContainedTexture::from_bytes(state, &data)
}
