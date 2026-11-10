#!/usr/bin/env python3
"""Compare experimental Metal ITP intermediates with the Rust reference.

Run after building both native libraries: python tests/verify_metal_itp.py [sample.ARW]
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'base'))
import fused_rust as rust
import metal_itp as metal
import sony_itp


def check(raw, black, gains, scale, cfg):
    for stage in (0, 1, 2, 3, 4, 5, 6, 18, 19, 22, 23, 24, 25, 26, 27):
        expected = rust.itp_debug_tile(raw, black, gains, scale, stage, cfg)
        actual = metal.debug_stage(raw, black, gains, scale, cfg.params, stage)
        if not np.array_equal(expected, actual):
            diff = np.abs(expected - actual)
            raise AssertionError(f'{raw.shape} stage {stage}: {np.count_nonzero(expected != actual)} mismatches; max {diff.max()}')


def main():
    cfg = rust.itp_config()
    black = [512, 510, 511, 513]
    gains = [2.1, 1.0, 1.0, 1.6]
    scale = 32768
    rng = np.random.default_rng(42)
    for shape in ((12, 12), (31, 32), (32, 31), (129, 130), (1048, 1048)):
        check(rng.integers(0, 16384, shape, dtype=np.uint16), black, gains, scale, cfg)
        print(f'{shape}: available stages identical')
    black, gains, scale = [512, 500, 520, 510], [2735, 1056, 1056, 1460], 256
    for shape in ((31, 32), (32, 31), (157, 262), (1027, 1051)):
        raw = rng.integers(0, 65536, shape, dtype=np.uint16)
        expected = rust.itp_full_image(raw, black, gains, scale, cfg, tile=128, halo=12, threads=6)
        actual = metal.full_image(raw, black, gains, scale, cfg.params, tile=128, halo=12)
        for x, y in zip(expected, actual):
            diff = np.abs(x.astype(np.int32) - y.astype(np.int32))
            if diff.max() > 1:
                raise AssertionError(f'{shape} full image: max RGB difference {diff.max()}')
        print(f'{shape}: tiled RGB within 1 unit')
    # Compare the batched GPU driver with the original tile-at-a-time path.
    raw = rng.integers(0, 16384, (157, 262), dtype=np.uint16)
    halo, tile = 12, 128
    padded = raw[np.ix_(sony_itp._edge_indices(raw.shape[0], halo),
                        sony_itp._edge_indices(raw.shape[1], halo))]
    old = [np.empty(raw.shape, dtype=np.uint16) for _ in range(3)]
    for y in range(0, raw.shape[0], tile):
        for x in range(0, raw.shape[1], tile):
            h, w = min(tile, raw.shape[0] - y), min(tile, raw.shape[1] - x)
            planes = metal.process_tile(padded[y:y+h+2*halo, x:x+w+2*halo], black, gains, scale, cfg.params)
            for dst, src in zip(old, planes):
                dst[y:y+h, x:x+w] = src[halo:halo+h, halo:halo+w]
    for expected, actual in zip(old, metal.full_image(raw, black, gains, scale, cfg.params, tile=tile, halo=halo)):
        if not np.array_equal(expected, actual):
            raise AssertionError("batched and original Metal tile outputs differ")
    print('batched and original Metal tile outputs identical')
    if len(sys.argv) > 1:
        import rawpy
        import arw_looks
        import sony_decode as decode
        import sony_frontend as front
        import develop_settings as settings
        path = sys.argv[1]
        main_tags = arw_looks.main_tags(path)
        st = settings.effective_settings(path)
        black = decode._ints(main_tags['0x7310'])
        levels = decode._ints(main_tags['0x7313'])
        scale = int(main_tags['0x7311'])
        with rawpy.imread(path) as image:
            raw = np.array(image.raw_image_visible[1000:2048, 1000:2048], copy=True)
        gains = front.wb_gains(levels, int(main_tags['0x7809']), int(st.get(settings.WB_TINT, 0)),
                               int(st.get(settings.EXPOSURE, 0)), mode=int(st.get(settings.WB_MODE, 1)),
                               temperature=st.get(settings.WB_TEMPERATURE, 6000),
                               temp_table=front.wb_temperature_table(main_tags),
                               presets=front.wb_presets(main_tags), rb_factor=int(st.get(settings.RB_FACTOR, 0)))
        check(raw, black, gains, scale, cfg)
        print(f'RAW {raw.shape}: available stages identical')
        expected = rust.itp_process_tile(raw, black, gains, scale, cfg)
        actual = metal.process_tile(raw, black, gains, scale, cfg.params)
        for name, x, y in zip('RGB', expected, actual):
            d = np.abs(x.astype(np.int32) - y.astype(np.int32))
            print(f'RAW tile {name}: different {np.count_nonzero(d)}, max {d.max()}')
            if d.max() > 1:
                raise AssertionError(f'RAW tile {name}: max difference {d.max()}')
        for name, run in [('Rust', lambda: rust.itp_debug_tile(raw, black, gains, scale, 6, cfg)),
                          ('Metal', lambda: metal.debug_stage(raw, black, gains, scale, cfg.params, 6))]:
            samples = []
            for _ in range(5):
                start = time.perf_counter(); run(); samples.append(time.perf_counter() - start)
            print(f'{name} stages 0–6 median {np.median(samples)*1000:.2f} ms')


if __name__ == '__main__':
    main()
