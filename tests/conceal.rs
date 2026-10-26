//! `Decoder::conceal_into` / `conceal` across codecs.
#![cfg(feature = "std")]

use audio_codec::{CodecType, create_decoder, create_encoder};

/// A frame of a continuous 150 Hz tone.
fn tone(rate: u32, frame_len: usize, frame: usize) -> Vec<i16> {
    (0..frame_len)
        .map(|i| {
            let t = (frame * frame_len + i) as f32 / rate as f32;
            (8000.0 * (2.0 * std::f32::consts::PI * 150.0 * t).sin()) as i16
        })
        .collect()
}

fn rms(s: &[i16]) -> f64 {
    (s.iter().map(|x| (*x as f64).powi(2)).sum::<f64>() / s.len() as f64).sqrt()
}

fn max_step(s: &[i16]) -> i32 {
    s.windows(2)
        .map(|w| (w[1] as i32 - w[0] as i32).abs())
        .max()
        .unwrap_or(0)
}

/// Decode a tone, lose one frame, conceal it, decode on: the concealment
/// continues the tone and the audio around it has no step.
#[cfg(feature = "plc")]
fn conceals_a_tone(codec: CodecType, rate: u32, frame_len: usize) {
    let mut enc = create_encoder(codec);
    let mut dec = create_decoder(codec);
    assert!(dec.enable_concealment(), "{codec:?}");
    let mut played = vec![];
    for f in 0..8 {
        played.extend(dec.decode(&enc.encode(&tone(rate, frame_len, f))));
    }
    // frame 8 is lost
    enc.encode(&tone(rate, frame_len, 8));
    let concealed = dec.conceal(frame_len);
    assert_eq!(concealed.len(), frame_len, "{codec:?}");
    assert!(
        rms(&concealed) > 0.5 * rms(&tone(rate, frame_len, 8)),
        "{codec:?} concealment too quiet: rms {}",
        rms(&concealed)
    );
    played.extend(&concealed);
    played.extend(dec.decode(&enc.encode(&tone(rate, frame_len, 9))));
    // the tone's own steepest step is ~470 at 16 kHz, ~940 at 8 kHz; a gap
    // or a misaligned seam would jump by thousands
    let around = &played[played.len() - 3 * frame_len..];
    assert!(
        max_step(around) < 2500,
        "{codec:?} step {} around the loss",
        max_step(around)
    );
}

#[cfg(feature = "plc")]
#[test]
fn pcma_conceals() {
    conceals_a_tone(CodecType::PCMA, 8000, 160);
}

#[cfg(feature = "plc")]
#[test]
fn pcmu_conceals() {
    conceals_a_tone(CodecType::PCMU, 8000, 160);
}

#[cfg(feature = "plc")]
#[test]
fn g722_conceals() {
    conceals_a_tone(CodecType::G722, 16000, 320);
}

#[cfg(feature = "plc")]
#[test]
fn long_loss_fades_to_silence() {
    for (codec, frame_len) in [(CodecType::PCMA, 160), (CodecType::G722, 320)] {
        let rate = if codec == CodecType::G722 {
            16000
        } else {
            8000
        };
        let mut enc = create_encoder(codec);
        let mut dec = create_decoder(codec);
        dec.enable_concealment();
        for f in 0..8 {
            dec.decode(&enc.encode(&tone(rate, frame_len, f)));
        }
        let frames: Vec<_> = (0..4).map(|_| dec.conceal(frame_len)).collect();
        assert!(rms(&frames[1]) < rms(&frames[0]), "{codec:?}");
        assert!(
            frames[3].iter().all(|s| *s == 0),
            "{codec:?} not silent at 60 ms"
        );
    }
}

#[cfg(feature = "plc")]
#[test]
fn concealment_before_any_audio_is_silence() {
    // a stream that starts with a loss: silence until there is audio to
    // conceal from, on every call (the history fills with that silence)
    for (codec, frame_len) in [(CodecType::PCMA, 160), (CodecType::G722, 320)] {
        let mut dec = create_decoder(codec);
        dec.enable_concealment();
        for _ in 0..6 {
            let out = dec.conceal(frame_len);
            assert_eq!(out.len(), frame_len, "{codec:?}");
            assert!(out.iter().all(|s| *s == 0), "{codec:?}");
        }
    }
}

#[cfg(feature = "plc")]
#[test]
fn empty_packet_does_not_end_a_loss() {
    let mut enc = create_encoder(CodecType::PCMA);
    let mut dec = create_decoder(CodecType::PCMA);
    dec.enable_concealment();
    for f in 0..8 {
        dec.decode(&enc.encode(&tone(8000, 160, f)));
    }
    let first = dec.conceal(160);
    let second = dec.conceal(160);
    assert!(dec.decode(&[]).is_empty());
    // carries on fading rather than restarting at full level
    let third = dec.conceal(160);
    assert!(rms(&second) < rms(&first));
    assert!(
        rms(&third) < rms(&second),
        "loss restarted after an empty packet"
    );
}

#[cfg(feature = "plc")]
#[test]
fn g722_recovers_after_a_long_loss() {
    let mut enc = create_encoder(CodecType::G722);
    let mut dec = create_decoder(CodecType::G722);
    dec.enable_concealment();
    for f in 0..8 {
        dec.decode(&enc.encode(&tone(16000, 320, f)));
    }
    // 200 ms lost: faded out long before it ends
    for f in 8..18 {
        enc.encode(&tone(16000, 320, f));
        assert_eq!(dec.conceal(320).len(), 320);
    }
    // the tone comes back at its level within a few packets
    let mut last = vec![];
    for f in 18..24 {
        last = dec.decode(&enc.encode(&tone(16000, 320, f)));
    }
    assert!(
        rms(&last) > 0.7 * rms(&tone(16000, 320, 23)),
        "rms {} after recovery",
        rms(&last)
    );
}

#[cfg(feature = "g729")]
#[test]
fn decoders_without_concealment_write_nothing() {
    let mut dec = create_decoder(CodecType::G729);
    let mut out = [1i16; 80];
    assert_eq!(dec.conceal_into(80, &mut out), Ok(0));
    assert!(dec.conceal(80).is_empty());
}

/// Without enabling, or without the `plc` feature, G.711 and G.722 leave
/// concealment to the caller.
#[test]
fn concealment_is_off_until_enabled() {
    for (codec, frame_len) in [
        (CodecType::PCMA, 160),
        (CodecType::PCMU, 160),
        (CodecType::G722, 320),
    ] {
        let rate = if codec == CodecType::G722 {
            16000
        } else {
            8000
        };
        let mut enc = create_encoder(codec);
        let mut dec = create_decoder(codec);
        dec.decode(&enc.encode(&tone(rate, frame_len, 0)));
        let mut out = vec![1i16; frame_len];
        assert_eq!(dec.conceal_into(frame_len, &mut out), Ok(0), "{codec:?}");
        assert!(
            out.iter().all(|s| *s == 1),
            "{codec:?} wrote while disabled"
        );
    }
}

#[cfg(feature = "plc")]
#[test]
fn disabled_concealment_costs_a_pointer() {
    use audio_codec::pcma::PcmaDecoder;
    use audio_codec::pcmu::PcmuDecoder;
    assert_eq!(
        std::mem::size_of::<PcmaDecoder>(),
        std::mem::size_of::<usize>()
    );
    assert_eq!(
        std::mem::size_of::<PcmuDecoder>(),
        std::mem::size_of::<usize>()
    );
}
