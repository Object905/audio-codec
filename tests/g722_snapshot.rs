//! G.722 encoder bit-exactness snapshot.
//!
//! Encodes a deterministic signal and pins the exact output bytes, so any
//! encoder change (e.g. the linear-search → binary-search quantiser tweak)
//! that alters the bitstream fails loudly.

use audio_codec::Encoder;
use audio_codec::g722::G722Encoder;

fn signal() -> Vec<i16> {
    (0..320)
        .map(|i| ((((i * 97) % 512) as i32 - 256) * 100) as i16)
        .collect()
}

#[test]
fn g722_encode_is_bit_exact() {
    let mut enc = G722Encoder::new();
    let mut out = vec![0u8; 1024];
    let n = enc.encode_into(&signal(), &mut out).unwrap();
    let hex: String = out[..n].iter().map(|b| format!("{b:02x}")).collect();
    println!("G722_HEX n={n} {hex}");
    assert_eq!(n, 160);
    assert_eq!(hex, "3284208420848424b48d2308ae9d0625ad90230aadbc882a07902310ecb5cd2a099623196ab0d2280a7896c7f1a953260e719b49eba5dc23176cbdce320ace253e6ab0542a0b5b90c8f5a955271071974af4a67f24116d9f4f3b09cc283e6bb3572f0dd4252fe6ab7b261172954bfda87f26116c98d03f08cc2a3b6bbd52740acf2b3167b05e2a10759049dcaa7827106d984df5a67024136abad779098e2b74");
}
