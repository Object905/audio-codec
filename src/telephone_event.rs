//! RFC 4733 telephone-event (DTMF) payload helpers.
//!
//! The Encoder/Decoder move no audio samples — telephone-event is an event
//! stream, not a waveform; they exist so `CodecType::TelephoneEvent` slots
//! into codec tables.

use super::{CodecError, Decoder, Encoder, Sample};

pub const EVENT_CLOCK: u32 = 8000;
pub const DEFAULT_EVENT_MS: u32 = 100;
const INTERIM_STEP_MS: u64 = 50;
const DEFAULT_VOLUME: u8 = 10;

pub struct TelephoneEventDecoder {}

impl TelephoneEventDecoder {
    pub fn new() -> Self {
        Self {}
    }
}

impl Default for TelephoneEventDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl Decoder for TelephoneEventDecoder {
    fn decode_into(&mut self, _data: &[u8], _out: &mut [Sample]) -> Result<usize, CodecError> {
        Ok(0)
    }

    fn max_decode_samples(&self, _n_bytes: usize) -> usize {
        0
    }

    fn sample_rate(&self) -> u32 {
        8000
    }

    fn channels(&self) -> u16 {
        1
    }
}

pub struct TelephoneEventEncoder {}

impl TelephoneEventEncoder {
    pub fn new() -> Self {
        Self {}
    }
}

impl Default for TelephoneEventEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl Encoder for TelephoneEventEncoder {
    fn encode_into(&mut self, _samples: &[Sample], _out: &mut [u8]) -> Result<usize, CodecError> {
        Ok(0)
    }

    fn max_encode_bytes(&self, _n_samples: usize) -> usize {
        0
    }

    fn sample_rate(&self) -> u32 {
        8000
    }

    fn channels(&self) -> u16 {
        1
    }
}

/// Character → RFC 4733 event code (`0-9`, `*`=10, `#`=11, `A-D`=12-15).
pub fn digit_to_event(c: char) -> Option<u8> {
    match c {
        '0'..='9' => Some(c as u8 - b'0'),
        '*' => Some(10),
        '#' => Some(11),
        'A'..='D' | 'a'..='d' => Some(12 + (c.to_ascii_uppercase() as u8 - b'A')),
        _ => None,
    }
}

/// RFC 4733 event code → character.
pub fn event_to_digit(event: u8) -> char {
    match event {
        0..=9 => (b'0' + event) as char,
        10 => '*',
        11 => '#',
        12..=15 => (b'A' + event - 12) as char,
        _ => '?',
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DtmfEvent {
    pub event: u8,
    pub end: bool,
    pub volume: u8,
    pub duration: u16,
}

/// Parse a telephone-event payload; with multiple 4-byte blocks the last wins.
pub fn parse_telephone_event(data: &[u8]) -> Option<DtmfEvent> {
    if data.len() < 4 || data.len() % 4 != 0 {
        return None;
    }
    let off = (data.len() / 4 - 1) * 4;
    Some(DtmfEvent {
        event: data[off],
        end: (data[off + 1] & 0x80) != 0,
        volume: data[off + 1] & 0x3F,
        duration: ((data[off + 2] as u16) << 8) | (data[off + 3] as u16),
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DtmfPacket {
    pub payload: [u8; 4],
    pub marker: bool,
}

/// `event(8) | E(1) R(1) volume(6) | duration(16)`.
pub fn encode_event_payload(event: u8, end: bool, volume: u8, duration: u16) -> [u8; 4] {
    let flags = (if end { 0x80 } else { 0x00 }) | (volume & 0x3F);
    [event, flags, (duration >> 8) as u8, (duration & 0xFF) as u8]
}

/// Packet sequence for one digit: interims + End ×3, marker on the first.
/// `duration_ms`/`clock_rate` are untrusted — clamped to u16::MAX wire ticks.
pub fn dtmf_sequence(digit: u8, duration_ms: u32, clock_rate: u32) -> alloc::vec::Vec<DtmfPacket> {
    let total_samples = u64::from(duration_ms) * u64::from(clock_rate) / 1000;
    let total_duration = total_samples.min(u64::from(u16::MAX)) as u16;
    let step = ((INTERIM_STEP_MS * u64::from(clock_rate) / 1000).min(u64::from(u16::MAX))) as u16;

    let mut packets = alloc::vec::Vec::new();
    let mut first = true;
    let mut d = step;
    while d < total_duration {
        packets.push(DtmfPacket {
            payload: encode_event_payload(digit, false, DEFAULT_VOLUME, d),
            marker: first,
        });
        first = false;
        d = d.saturating_add(step);
    }
    for _ in 0..3 {
        packets.push(DtmfPacket {
            payload: encode_event_payload(digit, true, DEFAULT_VOLUME, total_duration),
            marker: first,
        });
        first = false;
    }
    packets
}

pub fn dtmf_press(digit: u8) -> alloc::vec::Vec<DtmfPacket> {
    dtmf_sequence(digit, DEFAULT_EVENT_MS, EVENT_CLOCK)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn digit_mapping() {
        assert_eq!(digit_to_event('0'), Some(0));
        assert_eq!(digit_to_event('9'), Some(9));
        assert_eq!(digit_to_event('*'), Some(10));
        assert_eq!(digit_to_event('#'), Some(11));
        assert_eq!(digit_to_event('A'), Some(12));
        assert_eq!(digit_to_event('d'), Some(15));
        assert_eq!(digit_to_event('x'), None);
    }

    #[test]
    fn reverse_mapping() {
        for c in "0123456789*#ABCD".chars() {
            let e = digit_to_event(c).unwrap();
            assert_eq!(event_to_digit(e), c);
        }
    }

    #[test]
    fn encode_payload_shape() {
        assert_eq!(
            encode_event_payload(5, true, 10, 1600),
            [5, 0x8A, 0x06, 0x40]
        );
    }

    #[test]
    fn parse_event() {
        let e = parse_telephone_event(&[5, 0x8A, 0x06, 0x40]).unwrap();
        assert_eq!(e.event, 5);
        assert!(e.end);
        assert_eq!(e.volume, 10);
        assert_eq!(e.duration, 1600);
    }

    #[test]
    fn parse_takes_last_block() {
        let e = parse_telephone_event(&[3, 0x0A, 0x01, 0x90, 5, 0x8A, 0x06, 0x40]).unwrap();
        assert_eq!(e.event, 5);
        assert!(e.end);
    }

    #[test]
    fn parse_rejects_malformed() {
        assert!(parse_telephone_event(&[1, 2, 3]).is_none());
        assert!(parse_telephone_event(&[]).is_none());
        assert!(parse_telephone_event(&[1, 2, 3, 4, 5]).is_none());
    }

    #[test]
    fn sequence_shape() {
        let packets = dtmf_sequence(5, 200, EVENT_CLOCK);
        assert!(packets.len() >= 4);
        assert!(packets[0].marker);
        assert!(!packets[1..].iter().any(|p| p.marker));
        assert!(packets.iter().all(|p| p.payload[0] == 5));
        let last = packets.last().unwrap();
        assert_eq!(last.payload[1] & 0x80, 0x80);
        let dur = ((last.payload[2] as u16) << 8) | last.payload[3] as u16;
        assert_eq!(dur, 1600);
    }

    #[test]
    fn sequence_is_loss_resilient() {
        let packets = dtmf_sequence(3, 100, EVENT_CLOCK);
        assert_eq!(
            packets.iter().filter(|p| p.payload[1] & 0x80 != 0).count(),
            3
        );
    }

    #[test]
    fn roundtrip_encode_parse() {
        for digit in 0u8..16 {
            let press = dtmf_press(digit);
            let e = parse_telephone_event(&press.last().unwrap().payload).unwrap();
            assert_eq!(e.event, digit);
            assert!(e.end);
        }
    }

    #[test]
    fn extreme_durations_do_not_overflow() {
        let seq = dtmf_sequence(1, u32::MAX, EVENT_CLOCK);
        let last = seq.last().unwrap();
        let dur = ((last.payload[2] as u16) << 8) | last.payload[3] as u16;
        assert_eq!(dur, u16::MAX);
    }
}
