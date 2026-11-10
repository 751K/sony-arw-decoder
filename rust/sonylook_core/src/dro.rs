//! ZcTaskVatr (DRO) per-pixel operator.
//!
//! Direct port of sony_dro.vatr (with log_lookup and grid_lookup).  The grid
//! itself is built by sony_dro.build_grid in Python (about 0.1 s) and passed in.
//! float32 operations keep the reference order; fused multiply-adds use the
//! hardware FMA, and the gain is float32(exp2(float64(e))) as in NumPy.

use std::slice;

#[repr(C)]
pub struct DroParams {
    pub org_x: i64,
    pub org_y: i64,
    pub cw: i64, // subpx_x * nsub
    pub ch: i64, // subpx_y * nsub
    pub nx: i64,
    pub ny: i64,
    pub log2_max: f32,
    pub strength: f32,
    pub k_up: f32,
    pub k_down: f32,
    pub scale: f32,
    pub w_grid: *const f32, // ny x nx x 14
    pub s_grid: *const f32, // ny x nx x 14
    pub curve: *const f32,  // 104
    pub log_t: *const f32,  // 2049
}

const NB: i64 = 14;
const COEF: [f32; 3] = [0.1495, 0.2935, 0.057];

#[inline(always)]
fn log_lookup(v: f32, t: &[f32]) -> f32 {
    if v >= 16384.0 {
        return (v as f64).log2() as f32;
    }
    if !(v >= 0.0) {
        return 0.0;
    }
    let q = v / 8.0;
    let i = q.trunc() as usize;
    let fr = q - q.trunc();
    let a = t[i];
    fr.mul_add(t[i + 1] - a, a)
}

/// FUN_1000fe7d4 along one axis: (i0, i1, d0, d1).
#[inline(always)]
fn axis(p: i64, c0: i64, cw: i64, n: i64) -> (i64, i64, i64, i64) {
    let (i0, i1) = if p < c0 {
        (0, 0)
    } else if p < c0 + (n - 1) * cw {
        let i = (p - c0).div_euclid(cw.max(1));
        (i, i + 1)
    } else {
        (n - 1, n - 1)
    };
    (i0, i1, (p - (c0 + i0 * cw)).abs(), (p - (c0 + i1 * cw)).abs())
}

struct Grid<'a> {
    p: &'a DroParams,
    w: &'a [f32],
    s: &'a [f32],
    curve: &'a [f32],
    log_t: &'a [f32],
}

// Read-only during the parallel loop (DroParams holds raw pointers).
unsafe impl Sync for Grid<'_> {}

impl Grid<'_> {
    #[inline(always)]
    fn get(a: &[f32], nx: i64, j: i64, i: i64, b: i64) -> f32 {
        let b = b.min(NB + 1);
        if b >= NB {
            0.0 // bins past the end read 0
        } else {
            a[((j * nx + i) * NB + b) as usize]
        }
    }

    /// One output pixel (r, g, b already halved and in float32).
    #[inline(always)]
    fn pixel(&self, x: i64, y: i64, v: [f32; 3]) -> [u16; 3] {
        let p = self.p;
        let yl = COEF[2].mul_add(v[2], COEF[1].mul_add(v[1], COEF[0] * v[0]));
        let yl = yl + yl;
        let l = log_lookup(yl, self.log_t); // K54 = 1
        let px = (p.scale * x as f32) as i64;
        let py = (p.scale * y as f32) as i64;
        let (i0, i1, dx0, dx1) = axis(px, p.org_x + (p.cw >> 1), p.cw, p.nx);
        let (j0, j1, dy0, dy1) = axis(py, p.org_y + (p.ch >> 1), p.ch, p.ny);
        let b = l.max(0.0).floor() as i64;
        let f = l - l.floor();
        let b1 = if b < NB - 2 { b + 1 } else { NB - 1 };
        let (sx, sy) = (dx0 + dx1, dy0 + dy1);
        let (fdx0, fdx1, fdy0, fdy1) = (dx0 as f32, dx1 as f32, dy0 as f32, dy1 as f32);
        let inv = if sx != 0 { 1.0f32 / sx as f32 } else { 0.0 };
        let sy_f = sy.max(1) as f32;
        let interp = |a: &[f32], bb: i64| -> f32 {
            let nx = p.nx;
            let a0 = Self::get(a, nx, j0, i0, bb);
            let a1 = Self::get(a, nx, j0, i1, bb);
            let a2 = Self::get(a, nx, j1, i0, bb);
            let a3 = Self::get(a, nx, j1, i1, bb);
            let (top, bot) = if sx != 0 {
                (a1.mul_add(fdx0, a0 * fdx1) * inv, a3.mul_add(fdx0, a2 * fdx1) * inv)
            } else {
                (a0, a2)
            };
            if sy != 0 {
                bot.mul_add(fdy0, top * fdy1) / sy_f
            } else {
                top
            }
        };
        let (w0, w1) = (interp(self.w, b), interp(self.w, b1));
        let (s0, s1) = (interp(self.s, b), interp(self.s, b1));
        let wv = w1.mul_add(f, (-f).mul_add(w0, w0));
        let sv = s1.mul_add(f, s0 * (1.0 - f));
        let local = if wv == 0.0 { p.log2_max } else { sv / wv };
        let c = self.curve;
        let idx = (local * 8.0).max(0.0) as i64;
        let cv = if idx < 0x67 {
            let ic = idx.min(0x66) as usize;
            let t = (ic as f32).mul_add(-0.125, local) * 8.0;
            t.mul_add(c[ic + 1] - c[ic], c[ic])
        } else {
            c[103]
        };
        let e = (l - local).mul_add(p.strength, -l) + cv;
        let gain = (e as f64).exp2() as f32;
        let k = if gain - 1.0 > 0.0 { p.k_up } else { p.k_down };
        let m = gain.mul_add(k, -k);
        let mut out = [0u16; 3];
        for c in 0..3 {
            let o = m.mul_add(v[c], v[c]).min(16383.0).max(0.0);
            out[c] = ((o as i64) << 1) as u16;
        }
        out
    }
}

/// DRO on three w x h uint16 planes whose pixel (0, 0) sits at image
/// coordinates (ox, oy).  Returns 0, or -1 on invalid arguments.
#[no_mangle]
pub unsafe extern "C" fn dro_vatr_u16(
    r: *const u16,
    g: *const u16,
    b: *const u16,
    w: usize,
    h: usize,
    ox: i64,
    oy: i64,
    params: *const DroParams,
    threads: usize,
    ro: *mut u16,
    go: *mut u16,
    bo: *mut u16,
) -> i32 {
    if params.is_null() || w == 0 || h == 0 {
        return -1;
    }
    let p = &*params;
    if p.nx < 1 || p.ny < 1 || p.w_grid.is_null() || p.s_grid.is_null() || p.curve.is_null() || p.log_t.is_null() {
        return -1;
    }
    let ncell = (p.nx * p.ny * NB) as usize;
    let grid = Grid {
        p,
        w: slice::from_raw_parts(p.w_grid, ncell),
        s: slice::from_raw_parts(p.s_grid, ncell),
        curve: slice::from_raw_parts(p.curve, 104),
        log_t: slice::from_raw_parts(p.log_t, 2049),
    };
    let n = w * h;
    let (r, g, b) = (slice::from_raw_parts(r, n), slice::from_raw_parts(g, n), slice::from_raw_parts(b, n));
    let (ro, go, bo) = (
        slice::from_raw_parts_mut(ro, n),
        slice::from_raw_parts_mut(go, n),
        slice::from_raw_parts_mut(bo, n),
    );
    let threads = if threads > 0 {
        threads
    } else {
        std::thread::available_parallelism().map(|v| v.get()).unwrap_or(4)
    }
    .clamp(1, h);
    let chunk = (h + threads - 1) / threads;
    let grid = &grid;
    std::thread::scope(|s| {
        for (j, ((cr, cg), cb)) in ro
            .chunks_mut(chunk * w)
            .zip(go.chunks_mut(chunk * w))
            .zip(bo.chunks_mut(chunk * w))
            .enumerate()
        {
            s.spawn(move || {
                let y0 = j * chunk;
                for (k, ((vr, vg), vb)) in cr.iter_mut().zip(cg.iter_mut()).zip(cb.iter_mut()).enumerate() {
                    let i = y0 * w + k;
                    let (x, y) = ((i % w) as i64 + ox, (i / w) as i64 + oy);
                    let v = [(r[i] >> 1) as f32, (g[i] >> 1) as f32, (b[i] >> 1) as f32];
                    let o = grid.pixel(x, y, v);
                    *vr = o[0];
                    *vg = o[1];
                    *vb = o[2];
                }
            });
        }
    });
    0
}
