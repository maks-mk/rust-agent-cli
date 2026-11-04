//! Count exactly the bytes serde_json would write, without allocating a buffer.
use serde::ser::SerializeMap;
use serde::{Serialize, Serializer};
use serde_json::{Map, Value};
use std::io::{self, Write};

#[derive(Default)]
struct ByteCount(usize);

impl Write for ByteCount {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.0 = self.0.saturating_add(bytes.len());
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

pub(super) fn json_len(value: &impl Serialize) -> usize {
    let mut count = ByteCount::default();
    // Values and the filtered map below cannot fail to serialize to this writer.
    serde_json::to_writer(&mut count, value).expect("JSON value serialization failed");
    count.0
}

pub(super) struct VisibleMessage<'a>(pub &'a Map<String, Value>);

impl Serialize for VisibleMessage<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(None)?;
        for (key, value) in self.0 {
            if key != "responses_output_items" {
                map.serialize_entry(key, value)?;
            }
        }
        map.end()
    }
}
