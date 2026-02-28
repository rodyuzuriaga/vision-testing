import { env, InferenceSession, Tensor } from 'onnxruntime-web';

// WASM files served by onnxruntime-web from CDN
env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.24.2/dist/';

export async function createSession() {
    return await InferenceSession.create('/yolo26s-seg.onnx', { executionProviders: ['wasm'] });
}

export async function detectImage(session: InferenceSession, imgUrl: string, confidenceThreshold = 0.25, iouThreshold = 0.70) {
    const img = new Image();
    img.src = imgUrl;
    await new Promise((resolve) => {
        img.onload = resolve;
    });

    const canvas = document.createElement('canvas');
    const w = 640;
    const h = 640;
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d')!;

    const scale = Math.min(w / img.width, h / img.height);
    const dw = Math.round(img.width * scale);
    const dh = Math.round(img.height * scale);
    const dx = Math.round((w - dw) / 2);
    const dy = Math.round((h - dh) / 2);

    ctx.fillStyle = 'rgb(114, 114, 114)'; // YOLO default padding (crucial for valid convolution on edges)
    ctx.fillRect(0, 0, w, h);
    ctx.drawImage(img, dx, dy, dw, dh);

    const imgData = ctx.getImageData(0, 0, w, h).data;
    const inputArray = new Float32Array(3 * w * h);

    for (let i = 0; i < w * h; i++) {
        inputArray[i] = imgData[i * 4] / 255.0;
        inputArray[w * h + i] = imgData[i * 4 + 1] / 255.0;
        inputArray[2 * w * h + i] = imgData[i * 4 + 2] / 255.0;
    }

    const tensor = new Tensor('float32', inputArray, [1, 3, w, h]);
    const preT = performance.now();
    const results = await session.run({ images: tensor });
    const postT = performance.now();

    const output0Name = session.outputNames[0];
    const out0 = results[output0Name].data as Float32Array;
    const dims0 = results[output0Name].dims;

    let detections = [];
    let isNmsModel = false;

    if (dims0.length === 3 && dims0[2] === 38) {
        isNmsModel = true;
        // NMS Export format: [1, num_detections, 38]
        // 38 = 4 (bbox) + 1 (score) + 1 (class_id) + 32 (mask_coeffs)
        const numDetections = dims0[1];
        for (let i = 0; i < numDetections; i++) {
            const x1 = out0[i * 38 + 0];
            const y1 = out0[i * 38 + 1];
            const x2 = out0[i * 38 + 2];
            const y2 = out0[i * 38 + 3];
            const score = out0[i * 38 + 4];
            const classId = out0[i * 38 + 5];

            if (score > confidenceThreshold && classId >= 0 && score <= 1.0) {
                const maskCoeffs = new Float32Array(32);
                for (let k = 0; k < 32; k++) {
                    maskCoeffs[k] = out0[i * 38 + 6 + k];
                }

                detections.push({
                    box: [
                        Math.max(0, (x1 - dx) / scale),
                        Math.max(0, (y1 - dy) / scale),
                        Math.min(img.width, (x2 - dx) / scale),
                        Math.min(img.height, (y2 - dy) / scale)
                    ],
                    rawBox: [x1, y1, x2, y2],
                    score: score,
                    class: Math.round(classId),
                    maskCoeffs
                });
            }
        }
    } else {
        // Standard YOLOv8-seg output is [1, 115+4+32=151, 8400]
        let numRows, numCols;
        let transposed = false;

        if (dims0.length === 3) {
            if (dims0[1] < dims0[2]) {
                numRows = dims0[1]; // 151
                numCols = dims0[2]; // 8400
                transposed = false;
            } else {
                numRows = dims0[2]; // 151
                numCols = dims0[1]; // 8400
                transposed = true;
            }
        } else {
            numRows = dims0[0];
            numCols = dims0[1];
            transposed = numRows > numCols;
        }

        const numClasses = 115;

        for (let c = 0; c < numCols; c++) {
            let maxClassScore = 0;
            let classId = -1;

            for (let j = 0; j < numClasses; j++) {
                const idx = transposed ? (c * (numClasses + 4 + 32) + (j + 4)) : ((j + 4) * numCols + c);
                const score = out0[idx];
                if (score > maxClassScore) {
                    maxClassScore = score;
                    classId = j;
                }
            }

            if (maxClassScore > confidenceThreshold) {
                const x_idx = transposed ? (c * (numClasses + 4 + 32) + 0) : (0 * numCols + c);
                const y_idx = transposed ? (c * (numClasses + 4 + 32) + 1) : (1 * numCols + c);
                const w_idx = transposed ? (c * (numClasses + 4 + 32) + 2) : (2 * numCols + c);
                const h_idx = transposed ? (c * (numClasses + 4 + 32) + 3) : (3 * numCols + c);

                const xc = out0[x_idx];
                const yc = out0[y_idx];
                const w_box = out0[w_idx];
                const h_box = out0[h_idx];

                const x1 = xc - w_box / 2;
                const y1 = yc - h_box / 2;
                const x2 = xc + w_box / 2;
                const y2 = yc + h_box / 2;

                const maskCoeffs = new Float32Array(32);
                for (let k = 0; k < 32; k++) {
                    const mk_idx = transposed ? (c * (numClasses + 4 + 32) + (numClasses + 4 + k)) : ((numClasses + 4 + k) * numCols + c);
                    maskCoeffs[k] = out0[mk_idx];
                }

                detections.push({
                    box: [
                        Math.max(0, (x1 - dx) / scale),
                        Math.max(0, (y1 - dy) / scale),
                        Math.min(img.width, (x2 - dx) / scale),
                        Math.min(img.height, (y2 - dy) / scale)
                    ],
                    rawBox: [x1, y1, x2, y2],
                    score: Math.min(1.0, maxClassScore),
                    class: classId,
                    maskCoeffs
                });
            }
        }
    }

    // NMS — skip for models already exported with nms=true
    let finalDetections;
    if (isNmsModel) {
        // Model already did NMS, just sort by score
        detections.sort((a, b) => b.score - a.score);
        finalDetections = detections.slice(0, 50);
    } else {
        detections.sort((a, b) => b.score - a.score);
        const nmsDetections = [];
        while (detections.length > 0) {
            const best = detections[0];
            nmsDetections.push(best);
            detections.splice(0, 1);

            detections = detections.filter(det => {
                const iou = getIoU(best.box, det.box);
                return iou < iouThreshold;
            });
        }
        finalDetections = nmsDetections.slice(0, 50);
    }

    let out1: Float32Array | null = null;
    if (session.outputNames.length > 1) {
        const output1Name = session.outputNames[1];
        out1 = results[output1Name].data as Float32Array;
    }

    if (out1) {
        for (let i = finalDetections.length - 1; i >= 0; i--) {
            const d = finalDetections[i];

            const sx = Math.max(0, Math.floor(d.rawBox[0] / 4));
            const sy = Math.max(0, Math.floor(d.rawBox[1] / 4));
            const ex = Math.min(160, Math.ceil(d.rawBox[2] / 4));
            const ey = Math.min(160, Math.ceil(d.rawBox[3] / 4));

            const mask = new Uint8Array(160 * 160);
            for (let y = sy; y <= ey; y++) {
                for (let x = sx; x <= ex; x++) {
                    if (y >= 160 || x >= 160) continue;
                    let val = 0;
                    for (let k = 0; k < 32; k++) {
                        val += d.maskCoeffs[k] * out1[k * 160 * 160 + y * 160 + x];
                    }
                    if (val > 0) {
                        mask[y * 160 + x] = 1;
                    }
                }
            }

            const contour = getContour(mask, 160, 160);

            // Map the contour normalized to exactly the original image bounds [0-1]
            d.segments = {
                x: contour.x.map((cx: number) => Math.max(0, Math.min(1, (cx * 640 - dx) / dw))),
                y: contour.y.map((cy: number) => Math.max(0, Math.min(1, (cy * 640 - dy) / dh)))
            };
        }
    }

    // Format the bounding boxes to normalized output just in case
    for (let d of finalDetections) {
        d.box = {
            x1: Math.max(0, d.box[0] / img.width),
            y1: Math.max(0, d.box[1] / img.height),
            x2: Math.min(1, d.box[2] / img.width),
            y2: Math.min(1, d.box[3] / img.height)
        };
    }

    return {
        detections: finalDetections,
        time: postT - preT
    };
}

function hslToRgb(h: number, s: number, l: number) {
    let r, g, b;
    if (s === 0) {
        r = g = b = l;
    } else {
        const hue2rgb = (p: number, q: number, t: number) => {
            if (t < 0) t += 1;
            if (t > 1) t -= 1;
            if (t < 1 / 6) return p + (q - p) * 6 * t;
            if (t < 1 / 2) return q;
            if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
            return p;
        };
        const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
        const p = 2 * l - q;
        r = hue2rgb(p, q, h + 1 / 3);
        g = hue2rgb(p, q, h);
        b = hue2rgb(p, q, h - 1 / 3);
    }
    return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)];
}

function getContour(mask: Uint8Array, w: number, h: number) {
    // Step 1: Find all boundary pixels and compute centroid
    const boundary: [number, number][] = [];
    let cx = 0, cy = 0, totalMask = 0;

    for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
            if (mask[y * w + x] !== 1) continue;
            cx += x; cy += y; totalMask++;

            // A pixel is on the boundary if any 4-connected neighbor is 0 or out of bounds
            const isBoundary =
                x === 0 || x === w - 1 || y === 0 || y === h - 1 ||
                mask[y * w + (x - 1)] === 0 ||
                mask[y * w + (x + 1)] === 0 ||
                mask[(y - 1) * w + x] === 0 ||
                mask[(y + 1) * w + x] === 0;

            if (isBoundary) {
                boundary.push([x, y]);
            }
        }
    }

    if (totalMask === 0 || boundary.length === 0) return { x: [], y: [] };
    if (boundary.length <= 2) {
        return { x: boundary.map(p => p[0] / w), y: boundary.map(p => p[1] / h) };
    }

    cx /= totalMask;
    cy /= totalMask;

    // Step 2: Sort boundary pixels by angle from centroid (radial ordering)
    boundary.sort((a, b) => {
        const angleA = Math.atan2(a[1] - cy, a[0] - cx);
        const angleB = Math.atan2(b[1] - cy, b[0] - cx);
        return angleA - angleB;
    });

    // Step 3: Subsample to keep ~60-100 points for a clean polygon
    const maxPts = Math.min(boundary.length, 100);
    const step = Math.max(1, Math.floor(boundary.length / maxPts));
    const sampled: [number, number][] = [];
    for (let i = 0; i < boundary.length; i += step) {
        sampled.push(boundary[i]);
    }

    // Step 4: Light smoothing pass (moving average window=2)
    const smoothed: [number, number][] = [];
    const win = 2;
    for (let i = 0; i < sampled.length; i++) {
        let sx = 0, sy = 0, c = 0;
        for (let j = -win; j <= win; j++) {
            const idx = (i + j + sampled.length) % sampled.length;
            sx += sampled[idx][0];
            sy += sampled[idx][1];
            c++;
        }
        smoothed.push([sx / c, sy / c]);
    }

    return {
        x: smoothed.map(p => p[0] / w),
        y: smoothed.map(p => p[1] / h)
    };
}

function getIoU(box1: number[], box2: number[]) {
    const x1 = Math.max(box1[0], box2[0]);
    const y1 = Math.max(box1[1], box2[1]);
    const x2 = Math.min(box1[2], box2[2]);
    const y2 = Math.min(box1[3], box2[3]);
    const width = Math.max(0, x2 - x1);
    const height = Math.max(0, y2 - y1);
    const inter = width * height;
    const area1 = (box1[2] - box1[0]) * (box1[3] - box1[1]);
    const area2 = (box2[2] - box2[0]) * (box2[3] - box2[1]);
    return inter / (area1 + area2 - inter);
}
