//! Exact rustpbx FileTrack → next_audio_sample pipeline
//! Reads normal.wav (G.711 A-law 8k mono), decodes, resamples 8k→48k,
//! opus-encodes via the SAME frame-pull logic as rustpbx,
//! outputs a playable Ogg/Opus file + WAV for comparison.
//!
//! Run: cargo run --release --example filetrack_test -- normal.wav /tmp/filetrack.opus
//!
//! Then play /tmp/filetrack.opus on your baresip device and compare
//! with what you hear in the actual call.

use std::env;
use std::fs::File;
use std::io::{BufWriter, Write};

use audio_codec::opus::{OpusApplication, OpusEncoder};
use audio_codec::{BoxedResampler, Encoder as _};

// ---------- PCM 16-bit WAV writer ----------
fn write_wav_mono_48k(path: &str, samples: &[i16]) {
    let mut w = BufWriter::new(File::create(path).unwrap());
    let ns = samples.len() as u32;
    write!(w, "RIFF").ok();
    w.write_all(&(36u32 + ns * 2).to_le_bytes()).ok();
    write!(w, "WAVEfmt ").ok();
    w.write_all(&16u32.to_le_bytes()).ok();
    w.write_all(&1u16.to_le_bytes()).ok();
    w.write_all(&1u16.to_le_bytes()).ok();
    w.write_all(&48000u32.to_le_bytes()).ok();
    w.write_all(&(48000u32 * 2).to_le_bytes()).ok();
    w.write_all(&2u16.to_le_bytes()).ok();
    w.write_all(&16u16.to_le_bytes()).ok();
    write!(w, "data").ok();
    w.write_all(&(ns * 2).to_le_bytes()).ok();
    for &s in samples {
        w.write_all(&s.to_le_bytes()).ok();
    }
    w.flush().unwrap();
}

// ---------- Ogg Opus muxer ----------
fn ogg_crc_table() -> [u32; 256] {
    let mut t = [0u32; 256];
    for n in 0..256u32 {
        let mut c = n << 24;
        for _ in 0..8 {
            if c & 0x8000_0000 != 0 {
                c = (c << 1) ^ 0x04c1_1db7;
            } else {
                c <<= 1;
            }
        }
        t[n as usize] = c;
    }
    t
}

struct OggOpusWriter<W: Write> {
    w: W,
    crc: [u32; 256],
    serial: u32,
    page_seq: u32,
}

impl<W: Write> OggOpusWriter<W> {
    fn new(w: W, serial: u32) -> Self {
        Self { w, crc: ogg_crc_table(), serial, page_seq: 0 }
    }

    fn page(&mut self, packets: &[&[u8]], granule: i64, header_type: u8) -> std::io::Result<()> {
        let mut segs: Vec<u8> = Vec::new();
        let mut payload: Vec<u8> = Vec::new();
        for &p in packets {
            payload.extend_from_slice(p);
            let mut remaining = p.len();
            while remaining >= 255 { segs.push(255); remaining -= 255; }
            segs.push(remaining as u8);
        }
        let n_seg = segs.len() as u8;
        let mut header = Vec::with_capacity(27 + segs.len() + payload.len());
        header.extend_from_slice(b"OggS");
        header.push(0);
        header.push(header_type);
        header.extend_from_slice(&granule.to_le_bytes());
        header.extend_from_slice(&self.serial.to_le_bytes());
        header.extend_from_slice(&self.page_seq.to_le_bytes());
        header.extend_from_slice(&[0, 0, 0, 0]);
        header.push(n_seg);
        header.extend_from_slice(&segs);
        header.extend_from_slice(&payload);
        let mut crc = 0u32;
        for &b in &header {
            crc = (crc << 8) ^ self.crc[(((crc >> 24) as u8) ^ b) as usize];
        }
        header[22..26].copy_from_slice(&crc.to_le_bytes());
        self.w.write_all(&header)?;
        self.page_seq += 1;
        Ok(())
    }

    fn write_headers(&mut self, channels: u8) -> std::io::Result<()> {
        // OpusHead
        let mut head = Vec::new();
        head.extend_from_slice(b"OpusHead");
        head.push(1);
        head.push(channels);
        head.extend_from_slice(&0u16.to_le_bytes()); // preskip 0
        head.extend_from_slice(&48000u32.to_le_bytes());
        head.extend_from_slice(&0i16.to_le_bytes());
        head.push(0);
        self.page(&[&head], 0, 0x02)?;
        // OpusTags
        let vendor = b"filetrack_test";
        let mut tags = Vec::new();
        tags.extend_from_slice(b"OpusTags");
        tags.extend_from_slice(&(vendor.len() as u32).to_le_bytes());
        tags.extend_from_slice(vendor);
        tags.extend_from_slice(&0u32.to_le_bytes());
        self.page(&[&tags], 0, 0)?;
        Ok(())
    }
}

fn rms(s: &[i16]) -> f64 {
    if s.is_empty() { return 0.0; }
    (s.iter().map(|&x| (x as f64).powi(2)).sum::<f64>() / s.len() as f64).sqrt()
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: {} <input.wav> <output.opus>", args[0]);
        return;
    }
    let in_path = &args[1];
    let out_path = &args[2];
    let wav_path = out_path.replace(".opus", ".wav");

    // ---- Read and parse WAV ----
    let d = std::fs::read(in_path).unwrap_or_else(|e| panic!("read {}: {}", in_path, e));
    assert!(d.starts_with(b"RIFF"), "not RIFF");
    assert!(&d[8..12] == b"WAVE", "not WAVE");

    // Find fmt + data chunks
    let mut fmt_tag = 0u16; let mut ch = 1u16; let mut sr = 8000u32; let mut bps = 8u16;
    let mut data = Vec::new();
    let mut i = 12usize;
    while i + 8 <= d.len() {
        let cid = &d[i..i + 4];
        let sz = u32::from_le_bytes([d[i + 4], d[i + 5], d[i + 6], d[i + 7]]) as usize;
        let body = &d[i + 8..i + 8 + sz.min(d.len() - i - 8)];
        if cid == b"fmt " {
            fmt_tag = u16::from_le_bytes([body[0], body[1]]);
            ch = u16::from_le_bytes([body[2], body[3]]);
            sr = u32::from_le_bytes([body[4], body[5], body[6], body[7]]);
            bps = u16::from_le_bytes([body[14], body[15]]);
        } else if cid == b"data" {
            data = body.to_vec();
        }
        i += 8 + sz + (sz & 1);
    }
    eprintln!("WAV: fmt_tag={} ch={} sr={}bps={}  data={} bytes", fmt_tag, ch, sr, bps, data.len());

    // ---- FileAudioSource decode_all (A-law: read 160-bytes chunks → decode → cache) ----
    let pcm_cache: Vec<i16> = match fmt_tag {
        6 => { // A-law
            let mut dec = audio_codec::pcma::PcmaDecoder::new();
            let mut pcm = Vec::new();
            let mut pos = 0;
            while pos < data.len() {
                let chunk = &data[pos..pos + (data.len() - pos).min(160)];
                pos += chunk.len();
                pcm.extend_from_slice(&dec.decode(chunk));
            }
            if ch > 1 {
                // sync mono to stereo (rustpbx mix_stereo_to_mono)
                pcm.chunks_exact(ch as usize).map(|c| ((c.iter().map(|&x| x as i32).sum::<i32>()) / ch as i32) as i16).collect()
            } else {
                pcm
            }
        }
        1 if bps == 16 => {
            let pcm: Vec<i16> = data.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]])).collect();
            if ch > 1 {
                pcm.chunks_exact(ch as usize).map(|c| ((c.iter().map(|&x| x as i32).sum::<i32>()) / ch as i32) as i16).collect()
            } else {
                pcm
            }
        }
        _ => panic!("unsupported format tag {} bits {}", fmt_tag, bps),
    };
    eprintln!(
        "decode_all: {} samples @ {} Hz  RMS {:.0}",
        pcm_cache.len(), sr, rms(&pcm_cache)
    );

    // ---- Rustpbx next_audio_sample + ResamplingAudioSource ----
    let target_rate = 48000u32;
    let pcm_samples_per_frame = (target_rate * 20 / 1000) as usize; // 960
    let mut rs = BoxedResampler::new(sr as usize, target_rate as usize).expect("resampler");

    let mut cache_pos = 0usize;
    let mut pcm_buf = vec![0i16; pcm_samples_per_frame]; // reused per-frame

    let mut enc = OpusEncoder::new_with_application(target_rate, 2, OpusApplication::Voip);
    enc.set_bitrate(64000);
    enc.set_complexity(5);
    enc.set_cbr(true);

    let mut opus_packets: Vec<Vec<u8>> = Vec::new();
    let mut frame_no = 0;

    loop {
        // ---- ResamplingAudioSource::read_samples ----
        let needed_source = (pcm_buf.len() as u64 * sr as u64).div_ceil(target_rate as u64) as usize;
        let mut intermediate = vec![0i16; needed_source];

        let read = {
            let remaining = pcm_cache.len() - cache_pos;
            if remaining == 0 { break; }
            let copy = remaining.min(needed_source);
            intermediate[..copy].copy_from_slice(&pcm_cache[cache_pos..cache_pos + copy]);
            cache_pos += copy;
            copy
        };

        let resampled = rs.resample(&intermediate[..read]);
        let copy_len = resampled.len().min(pcm_buf.len());
        pcm_buf[..copy_len].copy_from_slice(&resampled[..copy_len]);

        // ---- next_audio_sample: encoder.encode() ----
        let encoded = enc.encode(&pcm_buf[..copy_len]);
        let pcm_rms = rms(&pcm_buf[..copy_len]);

        if frame_no < 3 || frame_no % 50 == 0 {
            eprintln!("  frame {:>3}: read={} copy_len={} pcm_rms={:.0} opus_len={}",
                frame_no, read, copy_len, pcm_rms, encoded.len());
        }

        opus_packets.push(encoded);
        frame_no += 1;
    }
    eprintln!("total frames: {}", opus_packets.len());

    // ---- Decode back to PCM for the reference WAV (using audio-codec's opus-rs decoder) ----
    let mut dec = audio_codec::opus::OpusDecoder::new(48000, 2);
    use audio_codec::Decoder as _;
    let mut decoded_pcm: Vec<i16> = Vec::new();
    for p in &opus_packets {
        let samples = dec.decode(p);
        decoded_pcm.extend_from_slice(&samples);
    }
    eprintln!("decoded: {} samples  RMS {:.0}", decoded_pcm.len(), rms(&decoded_pcm));

    // ---- Write Ogg/Opus file ----
    {
        let f = File::create(out_path).unwrap();
        let mut mux = OggOpusWriter::new(BufWriter::new(f), 0x12345678);
        mux.write_headers(2).unwrap();
        let mut buf: Vec<Vec<u8>> = Vec::new();
        let mut segs = 0usize;
        let mut last_idx = 0;
        for (idx, p) in opus_packets.iter().enumerate() {
            let p_segs = p.len() / 255 + 1;
            if segs + p_segs > 240 {
                let refs: Vec<&[u8]> = buf.iter().map(|v| v.as_slice()).collect();
                mux.page(&refs, (960 * last_idx) as i64, 0).unwrap();
                buf.clear(); segs = 0;
            }
            buf.push(p.clone());
            segs += p_segs;
            last_idx = idx + 1;
        }
        let total = (960 * opus_packets.len()) as i64;
        let refs: Vec<&[u8]> = buf.iter().map(|v| v.as_slice()).collect();
        mux.page(&refs, total, 0x04).unwrap();
    }
    eprintln!("Wrote: {} (Ogg/Opus, {} frames)", out_path, opus_packets.len());

    // ---- Write WAV for reference ----
    write_wav_mono_48k(&wav_path, &decoded_pcm);
    eprintln!("Wrote: {} (decoded ref, RMS {:.0})", wav_path, rms(&decoded_pcm));
}
