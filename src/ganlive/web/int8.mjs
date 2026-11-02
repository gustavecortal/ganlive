// Expands a program's 8-bit weights (engine/int8.py) into the fp16 weight buffer its shaders
// read, as the bytes arrive, so no copy of the download is kept.
//
//   const expand = expander(program.int8.matrices, program.buffers.P.size, mappedRange);
//   for (const chunk of chunks) expand.push(chunk);   // expand.bytes: the 8-bit file's size

const word = new Float32Array(1);
const bits = new Uint32Array(word.buffer);

/** The fp16 bits of `x` rounded to f32 (as numpy multiplies an int8 by an f32 scale), then
 *  to fp16, to nearest even. */
export function half(x) {
  word[0] = x;
  const b = bits[0], sign = (b >>> 16) & 0x8000, m = b & 0x7fffff;
  const e = ((b >>> 23) & 0xff) - 112;
  if (e <= 0) {                                  // fp16 subnormal, or zero
    if (e < -10) return sign;
    const shift = 14 - e, v = (m | 0x800000) >>> shift, rest = (m | 0x800000) & ((1 << shift) - 1);
    const mid = 1 << (shift - 1);
    return sign | (rest > mid || (rest === mid && v & 1) ? v + 1 : v);
  }
  if (e >= 31) return sign | 0x7c00;
  const v = (e << 10) | (m >>> 13), rest = m & 0x1fff;
  return sign | (rest > 0x1000 || (rest === 0x1000 && v & 1) ? v + 1 : v);
}

/** Writes the fp16 blob of `size` bytes into `target` (an ArrayBuffer) from the 8-bit file
 *  pushed to it in order. `matrices`: `[offset, rows, cols, axis]` each, as int8.py lists. */
export function expander(matrices, size, target) {
  const out8 = new Uint8Array(target, 0, size), out16 = new Uint16Array(target, 0, size / 2);
  const pieces = [];
  let src = 0, at = 0;
  const raw = (n) => {
    if (n > 0) pieces.push({ start: src, end: src + 2 * n, dst: 2 * at });
    src += 2 * n;
  };
  for (const [start, rows, cols, axis] of matrices) {
    raw(start - at);
    const k = axis ? cols : rows, n = rows * cols;
    pieces.push({ start: src, end: src + 4 * k + n, dst: start, k, cols, axis, scales: new Uint8Array(4 * k) });
    src += 4 * k + n;
    src += (4 - (src % 4)) % 4;                  // each matrix ends on a 4-byte word
    at = start + n;
  }
  raw(size / 2 - at);

  let pos = 0, i = 0;
  return {
    bytes: src,
    /** The next bytes of the file. */
    push(chunk) {
      const end = pos + chunk.byteLength;
      if (end > src) throw new Error(`weights: more than the ${src} bytes the program expects`);
      for (; i < pieces.length && pieces[i].start < end; i++) {
        const p = pieces[i], a = Math.max(pos, p.start), b = Math.min(end, p.end);
        if (!p.k) out8.set(chunk.subarray(a - pos, b - pos), p.dst + a - p.start);
        else matrix(p, chunk, a, b);
        if (p.end > end) break;                  // the rest of it comes in the next chunk
      }
      pos = end;
    },
    done() {
      if (pos !== src) throw new Error(`weights: got ${pos} bytes, the program expects ${src}`);
    },
  };

  function matrix(p, chunk, a, b) {
    const values = p.start + 4 * p.k;
    if (a < values) {
      const e = Math.min(b, values);
      p.scales.set(chunk.subarray(a - pos, e - pos), a - p.start);
      a = e;
    }
    if (a >= b) return;
    const scale = p.scale ??= new Float32Array(p.scales.buffer);
    const q = new Int8Array(chunk.buffer, chunk.byteOffset + a - pos, b - a);
    const first = a - values, cols = p.cols, dst = p.dst + first;
    if (p.axis) {
      // Each column's 256 values, made once: four times faster than rounding every weight.
      const lut = p.lut ??= Uint16Array.from({ length: 256 * cols }, (_, v) => half(((v & 255) << 24 >> 24) * scale[v >> 8]));
      for (let j = 0, c = first % cols; j < q.length; j++, c = c + 1 === cols ? 0 : c + 1)
        out16[dst + j] = lut[(c << 8) | (q[j] & 255)];
    } else {
      for (let j = 0, r = Math.floor(first / cols), c = first % cols; j < q.length; j++) {
        out16[dst + j] = half(q[j] * scale[r]);
        if (++c === cols) { c = 0; r++; }
      }
    }
  }
}
