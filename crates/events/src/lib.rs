//! Canonical event types shared across all shrimply services.
//!
//! Consolidates QueuedEvent definitions from pneuma, stt, tts, video_generation,
//! sam2, and tracking_3d crates and Python services into a single schema.
//!
//! All services that emit or consume events should import from this crate
//! rather than defining their own QueuedEvent type.

use serde::{Deserialize, Serialize};
use std::time::SystemTime;

/// Canonical representation of a queued event.
///
/// All shrimply services emit events with these three pieces of information:
/// - A unique identifier for this event
/// - When the event was created (for ordering and metrics)
/// - The event payload (serialized as bytes for flexibility)
///
/// This consolidates the three incompatible signatures previously scattered
/// across services:
/// - pneuma, stt, video_generation: (event_id, timestamp: u64, payload: Vec<u8>)
/// - tts, sam2: (event_id, created_at: SystemTime, data: serde_json::Value)
/// - tracking_3d: (id, ts: i64, body: String)
///
/// By standardizing on SystemTime and Vec<u8>, we get:
/// - Precise timing that works across all platforms
/// - Flexible payload representation (can serialize/deserialize to JSON as needed)
/// - Type safety that prevents timing confusion (u64 vs i64 vs SystemTime)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueuedEvent {
    /// Unique identifier for this event instance.
    pub event_id: String,

    /// Timestamp when this event was created.
    pub created_at: SystemTime,

    /// Event payload as bytes. Services serialize their domain objects
    /// to bytes (or JSON then to bytes) for this field.
    pub payload: Vec<u8>,
}

impl QueuedEvent {
    /// Create a new queued event with the current system time.
    pub fn new(event_id: impl Into<String>, payload: Vec<u8>) -> Self {
        Self {
            event_id: event_id.into(),
            created_at: SystemTime::now(),
            payload,
        }
    }

    /// Create a queued event with an explicit timestamp.
    pub fn at_time(
        event_id: impl Into<String>,
        created_at: SystemTime,
        payload: Vec<u8>,
    ) -> Self {
        Self {
            event_id: event_id.into(),
       created_at,
            payload,
        }
    }

    /// Parse the payload as JSON. Returns None if payload is not valid JSON.
    pub fn payload_as_json(&self) -> Option<serde_json::Value> {
        serde_json::from_slice(&self.payload).ok()
    }

    /// Set the payload from a serializable object.
    pub fn with_json_payload<T: Serialize>(&mut self, value: &T) -> Result<(), serde_json::Error> {
        self.payload = serde_json::to_vec(value)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn creates_event_with_current_time() {
        let event = QueuedEvent::new("test-1", vec![1, 2, 3]);
        assert_eq!(event.event_id, "test-1");
        assert_eq!(event.payload, vec![1, 2, 3]);
        // created_at should be very recent
        let elapsed = event.created_at.elapsed().unwrap();
        assert!(elapsed.as_secs() < 1);
    }

    #[test]
    fn serializes_to_json() {
        let event = QueuedEvent::new("test-2", vec![42]);
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("test-2"));
        assert!(json.contains("payload"));
    }

    #[test]
    fn parses_json_payload() {
        let obj = serde_json::json!({"type": "frame", "index": 10});
        let mut event = QueuedEvent::new("test-3", vec![]);
        event.with_json_payload(&obj).unwrap();
        
        let parsed = event.payload_as_json().unwrap();
        assert_eq!(parsed["type"], "frame");
        assert_eq!(parsed["index"], 10);
    }
}
