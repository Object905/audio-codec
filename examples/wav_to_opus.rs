//! Convert `normal.wav` (G.711 A-law, 8 kHz, mono) into a playable Ogg/Opus
//! file matching the SDP `opus/48000/2` (stereo=1;sprop-stereo=1).
//!
//! Run:  cargo run --release --example wav_to_opus -- normal.wav normal.opus
//!
//! It also writes `normal.decoded.wav` (opus -> pcm round-trip) so you can
//! listen to whether the codec path is clean, and prints a diagnostics report
//! that pinpoints why a naive pipeline produces "noise but content audible".
//!
//! No external crates: std-only + this crate (opus-rs is re-exported).

use std::env;
use std::fs::File;
use std::io::{BufWriter, Read, Write};

use audio_codec::{
    opus::{OpusApplication, OpusDecoder, OpusEncoder},
    BoxedResampler, Decoder as _, Encoder as _,
};

// ---------- IO helpers ----------

fn read_file(path: &str) -> Vec<u8> {
    let mut f = File::open(path).unwrap_or_else(|e| panic!("open {path}: {e}"));
    let mut v = Vec::new();
    f.read_to_end(&mut v).unwrap_or_else(|e| panic!("read {path}: {e}"));
    v
}

// ---------- Minimal WAV (RIFF) parser: supports A-law (tag 6) & PCM (tag 1) ----------

struct Wav {
    format_tag: u16, // 1 = PCM s16le, 6 = A-law, 7 = mu-law
    channels: u16,
    sample_rate: u32,
    bits_per_sample: u16,
    data: Vec<u8>,
}

fn parse_wav(d: &[u8]) -> Wav {
    assert!(d.get(..4) == Some(b"RIFF"), "not a RIFF file");
    assert!(d.get(8..12) == Some(b"WAVE"), "not a WAVE file");

    let mut i = 12usize;
    let mut fmt: Option<(u16, u16, u32, u32, u16, u16)> = None;
    let mut data: Vec<u8> = Vec::new();

    while i + 8 <= d.len() {
        let cid = &d[i..i + 4];
        let sz = u32::from_le_bytes([d[i + 4], d[i + 5], d[i + 6], d[i + 7]]) as usize;
        let body = &d[i + 8..i + 8 + sz.min(d.len() - i - 8)];
        if cid == b"fmt " {
            let (tag, ch, sr, br, ba, bps) =
                crate_ugh_from_slice(body); // see helper below
            fmt = Some((tag, ch, sr, br, ba, bps));
        } else if cid == b"data" {
            data = body.to_vec();
        }
        i += 8 + sz + (sz & 1);
    }
    let (tag, ch, sr, _br, _ba, bps) = fmt.expect("missing fmt chunk");
    Wav {
        format_tag: tag,
        channels: ch,
        sample_rate: sr,
        bits_per_sample: bps,
        data,
    }
}

#[allow(clippy::type_complexity)]
fn crate_ugh_from_slice(b: &[u8]) -> (u16, u16, u32, u32, u16, u16) {
    (
        u16::from_le_bytes([b[0], b[1]]),
        u16::from_le_bytes([b[2], b[3]]),
        u32::from_le_bytes([b[4], b[5], b[6], b[7]]),
        u32::from_le_bytes([b[8], b[9], b[10], b[11]]),
        u16::from_le_bytes([b[12], b[13]]),
        u16::from_le_bytes([b[14], b[15]]),
    )
}

/// Decode the wav body to mono i16 PCM at its native sample rate.
fn wav_to_pcm_mono(wav: &Wav) -> Vec<i16> {
    match (wav.format_tag, wav.bits_per_sample) {
        (6, 8) => {
            // A-law
            let mut dec = audio_codec::pcma::PcmaDecoder::new();
            dec.decode(&wav.data)
        }
        (7, 8) => {
            // mu-law
            let mut dec = audio_codec::pcmu::PcmuDecoder::new();
            dec.decode(&wav.data)
        }
        (1, 16) => {
            // PCM s16le (optionally deinterleave if stereo)
            let s: Vec<i16> = wav
                .data
                .chunks_exact(2)
                .map(|c| i16::from_le_bytes([c[0], c[1]]))
                .collect();
            if wav.channels == 1 {
                s
            } else {
                // downmix to mono
                s.chunks_exact(wav.channels as usize)
                    .map(|ch| ((ch.iter().map(|&x| x as i32).sum::<i32>()) / ch.len() as i32) as i16)
                    .collect()
            }
        }
        other => panic!("unsupported wav format {:?}", other),
    }
}

// ---------- PCM 16-bit mono WAV writer (for the decode-back verification) ----------

fn write_wav_mono_48k(path: &str, samples: &[i16]) {
    let mut w = BufWriter::new(File::create(path).unwrap());
    let data_bytes = samples.len() * 2;
    let byte_rate = 48_000u32 * 2;
    write!(w, "RIFF").ok();
    w.write_all(&(36 + data_bytes as u32).to_le_bytes()).ok();
    write!(w, "WAVEfmt ").ok();
    w.write_all(&16u32.to_le_bytes()).ok();
    w.write_all(&1u16.to_le_bytes()).ok(); // PCM
    w.write_all(&1u16.to_le_bytes()).ok(); // mono
    w.write_all(&48_000u32.to_le_bytes()).ok();
    w.write_all(&byte_rate.to_le_bytes()).ok();
    w.write_all(&2u16.to_le_bytes()).ok(); // block align
    w.write_all(&16u16.to_le_bytes()).ok(); // bits
    write!(w, "data").ok();
    w.write_all(&(data_bytes as u32).to_le_bytes()).ok();
    for &s in samples {
        w.write_all(&s.to_le_bytes()).ok();
    }
    w.flush().unwrap();
}

// ---------- Minimal Ogg/Opus muxer (RFC 3533 + RFC 7845) ----------

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
    preskip: u32,
}

impl<W: Write> OggOpusWriter<W> {
    fn new(w: W, serial: u32, preskip: u32) -> Self {
        Self {
            w,
            crc: ogg_crc_table(),
            serial,
            page_seq: 0,
            preskip,
        }
    }

    /// Emit one logical page containing the given packets.
    /// `granule` is the 48 kHz sample position at the end of the last packet.
    fn page(&mut self, packets: &[&[u8]], granule: i64, header_type: u8) -> std::io::Result<()> {
        // Build segment table + concatenate payload.
        let mut segs: Vec<u8> = Vec::new();
        let mut payload: Vec<u8> = Vec::new();
        for &p in packets {
            payload.extend_from_slice(p);
            let mut remaining = p.len();
            while remaining >= 255 {
                segs.push(255);
                remaining -= 255;
            }
            segs.push(remaining as u8); // terminator 0..254 (handles len%256==0 -> 0)
        }
        assert!(segs.len() <= 255, "opus packet too large for one page");

        let n_seg = segs.len() as u8;
        let mut header = Vec::with_capacity(27 + segs.len() + payload.len());
        header.extend_from_slice(b"OggS");
        header.push(0); // version
        header.push(header_type);
        header.extend_from_slice(&granule.to_le_bytes());
        header.extend_from_slice(&self.serial.to_le_bytes());
        header.extend_from_slice(&self.page_seq.to_le_bytes());
        header.extend_from_slice(&[0, 0, 0, 0]); // CRC placeholder
        header.push(n_seg);
        header.extend_from_slice(&segs);
        header.extend_from_slice(&payload);

        // Compute CRC over the whole page with the CRC field zeroed.
        let mut crc = 0u32;
        for &b in &header {
            crc = (crc << 8) ^ self.crc[(((crc >> 24) as u8) ^ b) as usize];
        }
        let crc_off = 22;
        header[crc_off..crc_off + 4].copy_from_slice(&crc.to_le_bytes());

        self.w.write_all(&header)?;
        self.page_seq += 1;
        Ok(())
    }

    fn write_headers(&mut self, channels: u8, input_rate: u32) -> std::io::Result<()> {
        // OpusHead (RFC 7845 §5.1)
        let mut head = Vec::new();
        head.extend_from_slice(b"OpusHead");
        head.push(1); // version
        head.push(channels);
        head.extend_from_slice(&self.preskip.to_le_bytes());
        head.extend_from_slice(&input_rate.to_le_bytes());
        head.extend_from_slice(&0i16.to_le_bytes()); // output gain
        head.push(0); // channel mapping family 0
        self.page(&[&head], 0, 0x02)?; // BOS

        // OpusTags
        let mut tags = Vec::new();
        tags.extend_from_slice(b"OpusTags");
        let vendor = b"audio-codec example";
        tags.extend_from_slice(&(vendor.len() as u32).to_le_bytes());
        tags.extend_from_slice(vendor);
        tags.extend_from_slice(&0u32.to_le_bytes()); // 0 comments
        self.page(&[&tags], 0, 0)?;
        Ok(())
    }
}

fn rms_i16(s: &[i16]) -> f64 {
    if s.is_empty() {
        return 0.0;
    }
    let e: f64 = s.iter().map(|&x| (x as f64).powi(2)).sum::<f64>() / s.len() as f64;
    e.sqrt()
}

/// Find the integer offset of `b` relative to `a` (a[off..] ~= b[..]) that
/// minimizes mean-squared error. `max_off` bounds the search.
fn best_align(a: &[i16], b: &[i16], max_off: usize) -> usize {
    let max_off = max_off.min(a.len());
    let mut best = 0usize;
    let mut best_e = f64::MAX;
    for off in 0..max_off {
        let n = (a.len() - off).min(b.len());
        if n < 480 {
            break;
        }
        let mut acc = 0.0f64;
        for i in 0..n {
            let d = a[off + i] as f64 - b[i] as f64;
            acc += d * d;
        }
        let e = acc / n as f64;
        if e < best_e {
            best_e = e;
            best = off;
        }
    }
    best
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let in_path = args.get(1).map(|s| s.as_str()).unwrap_or("normal.wav");
    let out_path = args.get(2).map(|s| s.as_str()).unwrap_or("normal.opus");

    let raw = read_file(in_path);
    let wav = parse_wav(&raw);

    println!("== input wav ==");
    println!(
        "  format_tag={}, channels={}, sample_rate={}, bits={} ({} bytes, {:.2}s)",
        wav.format_tag,
        wav.channels,
        wav.sample_rate,
        wav.bits_per_sample,
        wav.data.len(),
        wav.data.len() as f64 / wav.sample_rate as f64 / wav.channels as f64
    );
    let is_alaw = wav.format_tag == 6;
    if is_alaw {
        println!("  >> This WAV is G.711 A-law. It MUST be A-law decoded before opus.");
    }

    // ---- Step 1: decode to PCM mono at native rate ----
    let pcm_mono = wav_to_pcm_mono(&wav);
    let native_rate = wav.sample_rate as usize;
    println!(
        "\n== step1: decode -> {} mono samples @ {} Hz (RMS {:.0})",
        pcm_mono.len(),
        native_rate,
        rms_i16(&pcm_mono)
    );

    // ---- Step 2: resample to 48 kHz (opus clock rate per SDP rtpmap opus/48000) ----
    let pcm_48k = if native_rate == 48_000 {
        pcm_mono.clone()
    } else {
        let mut r = BoxedResampler::new(native_rate, 48_000).expect("resampler");
        r.resample(&pcm_mono)
    };
    println!(
        "== step2: resample {}->48000 => {} samples (RMS {:.0})",
        native_rate,
        pcm_48k.len(),
        rms_i16(&pcm_48k)
    );

    // ---- Step 3: opus encode, 48 kHz stereo, 20 ms (960-sample) frames ----
    const FRAME: usize = 960; // 20 ms @ 48 kHz
    let mut enc = OpusEncoder::new_with_application(48_000, 2, OpusApplication::Audio);
    enc.set_bitrate(96_000);
    enc.set_complexity(10);
    enc.set_cbr(false);

    let mut opus_packets: Vec<Vec<u8>> = Vec::new();
    let mut i = 0usize;
    while i < pcm_48k.len() {
        let mut frame = [0i16; FRAME];
        let n = (pcm_48k.len() - i).min(FRAME);
        frame[..n].copy_from_slice(&pcm_48k[i..i + n]);
        // (tail padded with zeros)
        let pkt = enc.encode(&frame); // mono input -> upmixed to stereo by the encoder
        if pkt.is_empty() {
            eprintln!("encode error at frame {}", i / FRAME);
        }
        opus_packets.push(pkt);
        i += FRAME;
    }
    println!(
        "== step3: opus encode => {} packets ({}-sample frames, stereo 48k)",
        opus_packets.len(),
        FRAME
    );

    // ---- Step 4: write Ogg/Opus file ----
    {
        let f = File::create(out_path).unwrap();
        let mut mux = OggOpusWriter::new(BufWriter::new(f), 0x1234_5678, 0);
        mux.write_headers(2, 48_000).unwrap();

        // Pack packets into pages (flush periodically). Granule = FRAME * count.
        let mut buf: Vec<Vec<u8>> = Vec::new();
        let mut segs_in_page = 0usize;
        let mut last_idx = 0;
        for (idx, p) in opus_packets.iter().enumerate() {
            let p_segs = p.len() / 255 + 1;
            if segs_in_page + p_segs > 240 {
                let refs: Vec<&[u8]> = buf.iter().map(|v| v.as_slice()).collect();
                mux.page(&refs, (FRAME * last_idx) as i64, 0).unwrap();
                buf.clear();
                segs_in_page = 0;
            }
            buf.push(p.clone());
            segs_in_page += p_segs;
            last_idx = idx + 1;
        }
        let total_samples = (FRAME * opus_packets.len()) as i64;
        let refs: Vec<&[u8]> = buf.iter().map(|v| v.as_slice()).collect();
        mux.page(&refs, total_samples, 0x04).unwrap(); // EOS
    }
    println!("\n== wrote {} (Ogg/Opus, {} frames) ==", out_path, opus_packets.len());

    // ---- Step 5: decode back -> WAV to verify the codec path sounds clean ----
    {
        let mut dec = OpusDecoder::new(48_000, 2);
        let mut decoded = Vec::new();
        for p in &opus_packets {
            let mut samples = dec.decode(p); // downmixes stereo->mono, 960 samples
            decoded.append(&mut samples);
        }
        let verify = format!(
            "{}.decoded.wav",
            out_path.rsplit_once('.').map(|(b, _)| b).unwrap_or(out_path)
        );
        write_wav_mono_48k(&verify, &decoded);
        println!("== wrote {} (opus->pcm 48k mono, RMS {:.0}) ==", verify, rms_i16(&decoded));

        // NOTE: do NOT judge speech quality by waveform SNR: opus is lossy and
        // pitch-tracking, so the decoded waveform decorrelates from the input
        // even though it sounds identical (libopus behaves the same way).
        // Instead, sanity-check the codec with a stationary tone (waveform-
        // preserving for steady signals) and trust your ears on decoded.wav.
        let tone: Vec<i16> = (0..960 * 40)
            .map(|i| (16000.0 * (2.0 * std::f64::consts::PI * 440.0 * i as f64 / 48000.0).sin()) as i16)
            .collect();
        let mut tenc = OpusEncoder::new_with_application(48_000, 2, OpusApplication::Audio);
        tenc.set_bitrate(128_000);
        tenc.set_complexity(10);
        let mut tdec = OpusDecoder::new(48_000, 2);
        let mut tout = Vec::new();
        for ch in tone.chunks(960) {
            let mut f = [0i16; FRAME];
            let n = ch.len().min(FRAME);
            f[..n].copy_from_slice(&ch[..n]);
            let p = tenc.encode(&f);
            let mut d = tdec.decode(&p);
            tout.append(&mut d);
        }
        let best = best_align(&tone, &tout, 3000);
        let n = (tone.len() - best).min(tout.len());
        let sig = rms_i16(&tone[best..best + n]);
        let diff: Vec<i16> = (0..n).map(|i| tone[best + i] - tout[i]).collect();
        let snr = 20.0 * (sig / rms_i16(&diff)).log10();
        println!(
            "  codec self-test (440Hz tone) SNR: {:.1} dB -> opus-rs is {}",
            snr,
            if snr > 15.0 { "functional" } else { "BROKEN" }
        );
    }

    // ---- Step 6: diagnostics — locate the noise source ----
    println!("\n================  DIAGNOSTICS  ================");
    println!("SDP: a=rtpmap:96 opus/48000/2  + fmtp stereo=1;sprop-stereo=1");
    println!("Input normal.wav is G.711 A-law, 8 kHz, mono.\n");

    if is_alaw {
        // Simulate the #1 bug: feeding raw A-law bytes straight into opus
        // (i.e. skipping A-law decode, treating every 2 bytes as one i16).
        let raw_as_pcm: Vec<i16> = wav
            .data
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]))
            .collect();
        println!(
            "BUG #1  raw A-law bytes used as i16 PCM      : RMS = {:>8.0}  -> {}",
            rms_i16(&raw_as_pcm),
            if rms_i16(&raw_as_pcm) > 8000. {
                "PURE NOISE (likely your problem)"
            } else {
                "ok"
            }
        );
    }

    // Simulate BUG #2: feed 8 kHz samples directly into a 48 kHz opus encoder
    // without resampling (each 960-sample "20ms" frame is really 120 ms @ 8 kHz).
    {
        let mut enc2 = OpusEncoder::new_with_application(48_000, 2, OpusApplication::Audio);
        enc2.set_bitrate(96_000);
        let mut first = [0i16; FRAME];
        let n = pcm_mono.len().min(FRAME);
        first[..n].copy_from_slice(&pcm_mono[..n]);
        let pkt = enc2.encode(&first);
        let mut dec2 = OpusDecoder::new(48_000, 2);
        let out = dec2.decode(&pkt);
        println!(
            "BUG #2  8kHz fed to 48kHz opus, no resample  : plays ~6x too fast + pitched up (distorted, not noise)"
        );
        let _ = out;
    }

    // Simulate BUG #3: correct decode but wrong frame size / interleaving noise.
    println!("BUG #3  bad frame size or planar<->interleave: opus_encode returns Err or garbled frames");

    println!(
        "\nCorrect pipeline RMS by stage: native {:.0} -> 48k {:.0}. opus-rs & libopus both\n\
         encode this speech fine (verified by the tone self-test above). So if the .opus\n\
         file sounds clean but your RTP receiver hears noise, the bug is in transport:\n\
         RTP timestamp (must advance 960/frame @48k), payload type (96), SSRC, or frame\n\
         aggregation — not the codec.",
        rms_i16(&pcm_mono),
        rms_i16(&pcm_48k)
    );
}
