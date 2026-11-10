//! ZcTaskRawNR on the whole CFA image (RGGB at even/even).
//!
//! Direct port of sony_rawnr.rawnr with region = the whole image.  Integer
//! arithmetic in i64 as in NumPy; the final blend is one float32 FMA, which is
//! what the reference computes (`_fma32_trunc`).

use std::slice;

#[repr(C)]
pub struct NrParams {
    pub off: [i32; 3],
    pub gain: [i32; 3],
    pub lim: [i32; 3],
    pub wm: [i32; 3],
    pub s: f32,
}

#[inline(always)]
fn channel(x: usize, y: usize) -> usize {
    (y & 1) + (x & 1) // 0 R, 1 G, 2 B
}

const G_MEAN: [(isize, isize); 25] = [
    (0, 0), (-4, 0), (0, -4), (0, 4), (4, 0), (0, -2), (0, 2), (-2, 0), (2, 0),
    (-2, -2), (2, -2), (-2, 2), (2, 2), (-3, -1), (-3, 1), (-1, -3), (-1, -1), (-1, 1), (-1, 3),
    (1, -3), (1, -1), (1, 1), (1, 3), (3, -1), (3, 1),
];

fn split_rows(raw: &[u16], w: usize, h: usize, p: &NrParams, y0: usize, hp: &mut [i16], base: &mut [u16]) {
    let rows = hp.len() / w;
    for r in 0..rows {
        let y = y0 + r;
        for x in 0..w {
            let o = r * w + x;
            if x < 2 || x >= w - 2 || y < 2 || y >= h - 2 {
                hp[o] = 0;
                base[o] = 0;
                continue;
            }
            let i = (y * w + x) as isize;
            let at = |dy: isize, dx: isize| raw[(i + dy * w as isize + dx) as usize] as i64;
            let c = at(0, 0);
            let k = channel(x, y);
            let d2 = at(-2, -2) + at(-2, 2) + at(2, -2) + at(2, 2);
            let a2 = at(0, -2) + at(-2, 0) + at(0, 2) + at(2, 0);
            let hv = if k == 1 {
                let d1 = at(-1, -1) + at(-1, 1) + at(1, -1) + at(1, 1);
                (((24 * c - (d2 + 2 * a2)) - 3 * d1) * 36) >> 10
            } else {
                (12 * c - (d2 + 2 * a2)) >> 4
            };
            hp[o] = hv as i16; // _i16: wrap to 16 bits
            base[o] = (p.off[k] as i64 + c - hv).clamp(0, 0x7fff) as u16;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn filter_rows(
    raw: &[u16],
    hp: &[i16],
    base: &[u16],
    thr: &[i32],
    w: usize,
    h: usize,
    p: &NrParams,
    y0: usize,
    out: &mut [u16],
) {
    let rows = out.len() / w;
    for r in 0..rows {
        let y = y0 + r;
        for x in 0..w {
            let i = y * w + x;
            let o = r * w + x;
            if x < 10 || x >= w - 10 || y < 10 || y >= h - 10 {
                out[o] = raw[i];
                continue;
            }
            let ii = i as isize;
            let b = |dy: isize, dx: isize| base[(ii + dy * w as isize + dx) as usize] as i64;
            let k = channel(x, y);
            let b0 = b(0, 0);
            let t = thr[k * 0x8000 + b0 as usize] as i64;
            let t2 = t * t;
            let m = if k == 1 {
                (G_MEAN.iter().map(|&(dy, dx)| b(dy, dx)).sum::<i64>() * 0x29) >> 10
            } else {
                let mut s = 0i64;
                for dy in [-4isize, 0, 4] {
                    for dx in [-4isize, 0, 4] {
                        s += b(dy, dx);
                    }
                }
                (s * 0x72) >> 10
            };
            let wm = p.wm[k] as i64;
            let cp = (wm * b0 + (0x400 - wm) * m) >> 10;
            let (mut acc, mut n) = (b0, 1i64);
            for dy in [-8isize, -4, 0, 4, 8] {
                for dx in [-8isize, -4, 0, 4, 8] {
                    if dy == 0 && dx == 0 {
                        continue;
                    }
                    let v = b(dy, dx);
                    let d = cp - v;
                    if d * d <= t2 {
                        acc += v;
                        n += 1;
                    }
                }
            }
            let avg = acc.div_euclid(n); // floor division, acc >= 0
            let lim = p.lim[k] as i64;
            let mut tv = (p.gain[k] as i64 * hp[i] as i64) >> 8;
            if tv <= -lim {
                tv = -lim;
            }
            if tv > lim {
                tv = lim;
            }
            let ov = (tv - p.off[k] as i64 + avg).clamp(0, 0x3fff);
            let rv = raw[i] as i64;
            let v = ((ov - rv) as f32).mul_add(p.s, rv as f32);
            out[o] = v.trunc() as i64 as u16;
        }
    }
}

fn run_rows<F>(out: &mut [u16], w: usize, h: usize, threads: usize, f: F)
where
    F: Fn(usize, &mut [u16]) + Sync,
{
    let chunk = (h + threads - 1) / threads;
    std::thread::scope(|s| {
        for (j, c) in out.chunks_mut(chunk * w).enumerate() {
            let f = &f;
            s.spawn(move || f(j * chunk, c));
        }
    });
}

/// RawNR on a w x h uint16 raw.  thr: three 0x8000-entry threshold tables.
/// Returns 0, or -1 on invalid arguments.
#[no_mangle]
pub unsafe extern "C" fn rawnr_u16(
    raw: *const u16,
    w: usize,
    h: usize,
    params: *const NrParams,
    thr: *const i32,
    threads: usize,
    out: *mut u16,
) -> i32 {
    if raw.is_null() || params.is_null() || thr.is_null() || out.is_null() || w < 21 || h < 21 {
        return -1;
    }
    let p = &*params;
    let n = w * h;
    let raw = slice::from_raw_parts(raw, n);
    let thr = slice::from_raw_parts(thr, 3 * 0x8000);
    let out = slice::from_raw_parts_mut(out, n);
    let threads = if threads > 0 {
        threads
    } else {
        std::thread::available_parallelism().map(|v| v.get()).unwrap_or(4)
    }
    .clamp(1, h);
    let mut hp = vec![0i16; n];
    let mut base = vec![0u16; n];
    let chunk = (h + threads - 1) / threads;
    std::thread::scope(|s| {
        for ((j, a), b) in hp.chunks_mut(chunk * w).enumerate().zip(base.chunks_mut(chunk * w)) {
            s.spawn(move || split_rows(raw, w, h, p, j * chunk, a, b));
        }
    });
    let (hp, base) = (&hp[..], &base[..]);
    run_rows(out, w, h, threads, |y0, c| filter_rows(raw, hp, base, thr, w, h, p, y0, c));
    0
}
