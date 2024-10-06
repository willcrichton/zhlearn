use anyhow::Result;
use serde::{de::DeserializeOwned, Serialize};
use std::{
  fs::File,
  io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write},
  ops::Range,
  path::Path,
};

pub struct FileDbWriter {
  writer: BufWriter<File>,
  byte_pos: u64,
  buf: Vec<u8>,
}

impl FileDbWriter {
  pub fn new(path: impl AsRef<Path>) -> Result<Self> {
    Ok(FileDbWriter {
      writer: BufWriter::new(File::create(path)?),
      byte_pos: 0,
      buf: Vec::new(),
    })
  }

  pub fn write<T: Serialize>(&mut self, obj: &T) -> Result<Range<u64>> {
    self.buf.clear();
    serde_json::to_writer(&mut self.buf, obj)?;
    self.writer.write(&self.buf)?;

    let start = self.byte_pos;
    let len = u64::try_from(self.buf.len()).unwrap();
    let range = start..(start + len);
    self.byte_pos += len;

    Ok(range)
  }
}

pub struct FileDbReader {
  reader: BufReader<File>,
  buf: Vec<u8>,
}

impl FileDbReader {
  pub fn load(path: impl AsRef<Path>) -> Result<Self> {
    Ok(FileDbReader {
      reader: BufReader::new(File::open(path)?),
      buf: Vec::new(),
    })
  }

  pub fn read<T: DeserializeOwned>(&mut self, range: Range<u64>) -> Result<T> {
    self.reader.seek(SeekFrom::Start(range.start))?;
    self.buf.clear();
    (&mut self.reader)
      .take(range.end - range.start)
      .read_to_end(&mut self.buf)?;
    Ok(serde_json::from_slice(&self.buf)?)
  }
}
