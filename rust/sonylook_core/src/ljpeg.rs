//! Sony lossless compressed raw (TIFF compression 7).
//!
//! The CFA is stored as tiles; each tile is a lossless JPEG (SOF3) whose four
//! components are the 2x2 block at (2*row, 2*col): component c lands at
//! (2*row + c/2, 2*col + c%2).  This matches LibRaw's sony_ljpeg_load_raw and
//! ljpeg_row: predictor selection per ITU T.81 (first sample from
//! 2^(P-Pt-1), first row from the left, first column from above), Huffman
//! category 16 = -32768, 0xFF00 stuffing, and a marker ends the entropy data
//! (zeros afterwards).  Tiles are independent and decoded on several threads.

use std::slice;
use std::sync::atomic::{AtomicUsize, Ordering};

struct Huff {
    // Indexed by the next 16 bits: (code length << 8) | category, 0 = invalid.
    lut: Vec<u16>,
}

impl Huff {
    fn new(bits: &[u8; 16], vals: &[u8]) -> Option<Huff> {
        let mut lut = vec![0u16; 1 << 16];
        let mut code: u32 = 0;
        let mut k = 0usize;
        for len in 1..=16u32 {
            for _ in 0..bits[len as usize - 1] {
                let v = *vals.get(k)? as u16;
                if v > 16 || code >= (1 << len) {
                    return None;
                }
                let shift = 16 - len;
                let base = (code << shift) as usize;
                for e in &mut lut[base..base + (1usize << shift)] {
                    *e = ((len as u16) << 8) | v;
                }
                code += 1;
                k += 1;
            }
            code <<= 1;
        }
        Some(Huff { lut })
    }
}

struct Bits<'a> {
    data: &'a [u8],
    pos: usize,
    buf: u64,
    n: u32,
    end: bool,
}

impl<'a> Bits<'a> {
    #[inline(always)]
    fn fill(&mut self) {
        while self.n <= 56 {
            let mut b = 0u8;
            if !self.end && self.pos < self.data.len() {
                b = self.data[self.pos];
                self.pos += 1;
                if b == 0xff {
                    let next = if self.pos < self.data.len() { self.data[self.pos] } else { 0xd9 };
                    if next == 0 {
                        self.pos += 1;
                    } else {
                        self.end = true;
                        b = 0;
                    }
                }
            }
            self.buf |= (b as u64) << (56 - self.n);
            self.n += 8;
        }
    }
    #[inline(always)]
    fn peek16(&mut self) -> u32 {
        if self.n < 16 {
            self.fill();
        }
        (self.buf >> 48) as u32
    }
    #[inline(always)]
    fn skip(&mut self, k: u32) {
        self.buf <<= k;
        self.n -= k;
    }
    #[inline(always)]
    fn get(&mut self, k: u32) -> u32 {
        if k == 0 {
            return 0;
        }
        if self.n < k {
            self.fill();
        }
        let v = (self.buf >> (64 - k)) as u32;
        self.skip(k);
        v
    }
}

#[inline(always)]
fn diff(b: &mut Bits, h: &Huff) -> Option<i32> {
    let e = h.lut[b.peek16() as usize];
    if e == 0 {
        return None;
    }
    b.skip((e >> 8) as u32);
    let len = (e & 0xff) as u32;
    if len == 16 {
        return Some(-32768);
    }
    let v = b.get(len) as i32;
    Some(if len > 0 && v & (1 << (len - 1)) == 0 { v - (1 << len) + 1 } else { v })
}

struct Header {
    precision: u32,
    width: usize,
    height: usize,
    comps: usize,
    tables: [usize; 4],
    predictor: u32,
    pt: u32,
    entropy: usize,
}

fn be16(d: &[u8], i: usize) -> Option<usize> {
    Some(((*d.get(i)? as usize) << 8) | *d.get(i + 1)? as usize)
}

/// Parse markers up to SOS.  Returns the header and the Huffman tables.
fn parse(d: &[u8]) -> Option<(Header, Vec<Option<Huff>>)> {
    if be16(d, 0)? != 0xffd8 {
        return None;
    }
    let mut pos = 2;
    let mut huffs: Vec<Option<Huff>> = (0..4).map(|_| None).collect();
    let (mut precision, mut width, mut height, mut comps) = (0u32, 0usize, 0usize, 0usize);
    let mut ids = [0u8; 4];
    loop {
        while *d.get(pos)? == 0xff && *d.get(pos + 1)? == 0xff {
            pos += 1;
        }
        let marker = be16(d, pos)?;
        let len = be16(d, pos + 2)?;
        let seg = d.get(pos + 4..pos + 2 + len)?;
        match marker {
            0xffc3 => {
                precision = seg[0] as u32;
                height = be16(seg, 1)?;
                width = be16(seg, 3)?;
                comps = *seg.get(5)? as usize;
                if comps != 4 || precision < 2 || precision > 16 {
                    return None;
                }
                for c in 0..4 {
                    ids[c] = *seg.get(6 + 3 * c)?;
                    if *seg.get(7 + 3 * c)? != 0x11 {
                        return None; // only 1x1 sampling
                    }
                }
            }
            0xffc4 => {
                let mut i = 0;
                while i < seg.len() {
                    let tc_th = seg[i];
                    let mut bits = [0u8; 16];
                    bits.copy_from_slice(seg.get(i + 1..i + 17)?);
                    let n: usize = bits.iter().map(|&b| b as usize).sum();
                    let vals = seg.get(i + 17..i + 17 + n)?;
                    let th = (tc_th & 0x0f) as usize;
                    if th >= 4 {
                        return None;
                    }
                    huffs[th] = Some(Huff::new(&bits, vals)?);
                    i += 17 + n;
                }
            }
            0xffdd => {
                if be16(seg, 0)? != 0 {
                    return None; // restart intervals are not used by Sony
                }
            }
            0xffda => {
                let ns = *seg.first()? as usize;
                if ns != comps || comps == 0 {
                    return None;
                }
                let mut tables = [0usize; 4];
                for k in 0..ns {
                    let id = *seg.get(1 + 2 * k)?;
                    let c = ids.iter().position(|&v| v == id)?;
                    tables[c] = (*seg.get(2 + 2 * k)? >> 4) as usize;
                    if tables[c] >= 4 || huffs[tables[c]].is_none() {
                        return None;
                    }
                }
                let predictor = *seg.get(1 + 2 * ns)? as u32;
                let pt = (*seg.get(3 + 2 * ns)? & 0x0f) as u32;
                if !(1..=7).contains(&predictor) || pt >= precision {
                    return None;
                }
                let h = Header { precision, width, height, comps, tables, predictor, pt, entropy: pos + 2 + len };
                return Some((h, huffs));
            }
            0xffd8 | 0xffd9 => return None,
            _ => {}
        }
        pos += 2 + len;
    }
}

#[derive(Clone, Copy)]
struct OutPtr(*mut u16);
unsafe impl Send for OutPtr {}
unsafe impl Sync for OutPtr {}

/// Decode one tile into the raw image at (trow, tcol).
unsafe fn tile(d: &[u8], out: OutPtr, rw: usize, rh: usize, trow: usize, tcol: usize, tw: usize, th: usize) -> Option<()> {
    let (h, huffs) = parse(d)?;
    if 2 * h.width != tw || 2 * h.height != th {
        return None;
    }
    let mut bits = Bits { data: &d[h.entropy..], pos: 0, buf: 0, n: 0, end: false };
    let nc = h.comps;
    let w = h.width;
    let mut prev = vec![0u16; w * nc];
    let mut cur = vec![0u16; w * nc];
    let init = 1i32 << (h.precision - h.pt - 1);
    let tab: Vec<&Huff> = (0..nc).map(|c| huffs[h.tables[c]].as_ref().unwrap()).collect();
    for jrow in 0..h.height {
        for col in 0..w {
            for c in 0..nc {
                let dv = diff(&mut bits, tab[c])?;
                let i = col * nc + c;
                let pred: i32 = if jrow == 0 && col == 0 {
                    init
                } else if jrow == 0 {
                    cur[i - nc] as i32
                } else if col == 0 {
                    prev[i] as i32
                } else {
                    let (ra, rb, rc) = (cur[i - nc] as i32, prev[i] as i32, prev[i - nc] as i32);
                    match h.predictor {
                        1 => ra,
                        2 => rb,
                        3 => rc,
                        4 => ra + rb - rc,
                        5 => ra + ((rb - rc) >> 1),
                        6 => rb + ((ra - rc) >> 1),
                        _ => (ra + rb) >> 1,
                    }
                };
                cur[i] = (pred + dv) as u16;
            }
        }
        let row = trow + 2 * jrow;
        for col in 0..w {
            let x = tcol + 2 * col;
            for c in 0..nc {
                let (y, xx) = (row + c / 2, x + c % 2);
                if y < rh && xx < rw {
                    *out.0.add(y * rw + xx) = cur[col * nc + c] << h.pt;
                }
            }
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    Some(())
}

/// Decode all tiles of a compression-7 raw.  `data` is the whole file;
/// offsets/counts are TileOffsets/TileByteCounts.  Returns 0, -1 for bad
/// arguments, or 1 + the index of the first tile that failed to decode.
#[no_mangle]
pub unsafe extern "C" fn ljpeg_tiles_u16(
    data: *const u8,
    data_len: usize,
    offsets: *const u64,
    counts: *const u64,
    ntiles: usize,
    raw_width: usize,
    raw_height: usize,
    tile_width: usize,
    tile_height: usize,
    threads: usize,
    out: *mut u16,
) -> i32 {
    if data.is_null() || offsets.is_null() || counts.is_null() || out.is_null() || tile_width == 0
        || tile_height == 0 || raw_width == 0 || raw_height == 0
    {
        return -1;
    }
    let across = (raw_width + tile_width - 1) / tile_width;
    let down = (raw_height + tile_height - 1) / tile_height;
    if ntiles < across * down {
        return -1;
    }
    let data = slice::from_raw_parts(data, data_len);
    let offs = slice::from_raw_parts(offsets, ntiles);
    let cnts = slice::from_raw_parts(counts, ntiles);
    let ptr = OutPtr(out);
    let threads = if threads > 0 {
        threads
    } else {
        std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4)
    }
    .min(across * down)
    .max(1);
    let next = AtomicUsize::new(0);
    let failed = AtomicUsize::new(usize::MAX);
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| loop {
                let t = next.fetch_add(1, Ordering::Relaxed);
                if t >= across * down {
                    break;
                }
                let (o, n) = (offs[t] as usize, cnts[t] as usize);
                let ok = o.checked_add(n).filter(|&e| e <= data.len()).and_then(|e| {
                    tile(&data[o..e], ptr, raw_width, raw_height, (t / across) * tile_height,
                         (t % across) * tile_width, tile_width, tile_height)
                });
                if ok.is_none() {
                    failed.fetch_min(t, Ordering::Relaxed);
                }
            });
        }
    });
    match failed.load(Ordering::Relaxed) {
        usize::MAX => 0,
        t => (t + 1) as i32,
    }
}
