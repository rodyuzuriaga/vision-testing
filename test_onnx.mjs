import * as fs from 'fs';
import * as onnx from 'onnxruntime-node';

async function test() {
    const session = await onnx.InferenceSession.create('./public/yolo26s-seg.onnx');
    const inputArray = new Float32Array(3 * 640 * 640);
    // Let's create a red square at 100, 100, 50x50
    for (let i = 0; i < 640 * 640; i++) {
        const y = Math.floor(i / 640);
        const x = i % 640;
        if (x >= 100 && x < 150 && y >= 100 && y < 150) {
            inputArray[i] = 1.0;
        }
    }
    const tensor = new onnx.Tensor('float32', inputArray, [1, 3, 640, 640]);
    const results = await session.run({ images: tensor });
    const out0 = results[session.outputNames[0]].data;

    // 38 elements per detection
    console.log("First row:");
    const row = out0.slice(0, 38);
    console.log(row);
}
test().catch(console.error);
