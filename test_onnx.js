const fs = require('fs');
const onnx = require('onnxruntime-node');
const { createCanvas, loadImage } = require('canvas');

async function test() {
    console.log("Loading model...");
    const session = await onnx.InferenceSession.create('./public/yolo26s-seg.onnx');
    console.log("Input names:", session.inputNames);
    console.log("Output names:", session.outputNames);

    const canvas = createCanvas(640, 640);
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = 'red';
    ctx.fillRect(100, 100, 50, 50); // fake object
    const imgData = ctx.getImageData(0, 0, 640, 640).data;
    
    const inputArray = new Float32Array(3 * 640 * 640);
    for (let i = 0; i < 640 * 640; i++) {
        inputArray[i] = imgData[i * 4] / 255.0;
        inputArray[640 * 640 + i] = imgData[i * 4 + 1] / 255.0;
        inputArray[2 * 640 * 640 + i] = imgData[i * 4 + 2] / 255.0;
    }
    const tensor = new onnx.Tensor('float32', inputArray, [1, 3, 640, 640]);
    const results = await session.run({ images: tensor });
    
    const out0 = results[session.outputNames[0]];
    console.log("Out0 dims:", out0.dims);
    
    // Check first detection
    if (out0.dims.length === 3) {
        // [1, 115+4+32=151, 8400] or [1, 8400, 151]
        let numRows = out0.dims[1];
        let numCols = out0.dims[2];
        let transposed = numRows < numCols;
        console.log("Transposed:", transposed, "Rows:", numRows, "Cols:", numCols);
        
        const data = out0.data;
        let c = 0; // first anchor
        const x_idx = transposed ? (c * 151 + 0) : (0 * numCols + c);
        console.log("Anchor 0 x:", data[x_idx], "y:", data[transposed ? (c * 151 + 1) : (1 * numCols + c)], "w:", data[transposed ? (c * 151 + 2) : (2 * numCols + c)], "h:", data[transposed ? (c * 151 + 3) : (3 * numCols + c)]);
    }
}
test().catch(console.error);
