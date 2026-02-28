import * as fs from 'fs';
import * as onnx from 'onnxruntime-node';
import { createCanvas, loadImage } from 'canvas';

async function test() {
    const session = await onnx.InferenceSession.create('./public/yolo26s-seg.onnx');
    
    // Load banana image
    const bananaUrl = './public/dataset/banana_anthracnose_Bing_0007.jpg';
    if (!fs.existsSync(bananaUrl)) {
        console.error("Banana not found!");
        return;
    }
    const img = await loadImage(bananaUrl);
    
    const canvas = createCanvas(640, 640);
    const ctx = canvas.getContext('2d');
    
    const w = 640;
    const h = 640;
    const scale = Math.min(w / img.width, h / img.height);
    const dw = img.width * scale;
    const dh = img.height * scale;
    const dx = (w - dw) / 2;
    const dy = (h - dh) / 2;
    
    ctx.fillStyle = '#727272'; // Grey padding
    ctx.fillRect(0, 0, w, h);
    ctx.drawImage(img, dx, dy, dw, dh);
    
    const imgData = ctx.getImageData(0, 0, w, h).data;
    const inputArray = new Float32Array(3 * w * h);
    for (let i = 0; i < w * h; i++) {
        inputArray[i] = imgData[i * 4] / 255.0; // R
        inputArray[w * h + i] = imgData[i * 4 + 1] / 255.0; // G
        inputArray[2 * w * h + i] = imgData[i * 4 + 2] / 255.0; // B
    }
    
    const tensor = new onnx.Tensor('float32', inputArray, [1, 3, 640, 640]);
    const results = await session.run({ images: tensor });
    
    const out0 = results[session.outputNames[0]].data;
    
    const numDetections = 10;
    for (let i = 0; i < numDetections; i++) {
        const x1 = out0[i * 38 + 0];
        const y1 = out0[i * 38 + 1];
        const x2 = out0[i * 38 + 2];
        const y2 = out0[i * 38 + 3];
        const score = out0[i * 38 + 4];
        const classId = out0[i * 38 + 5];
        if (score > 0.05) {
            console.log(`Det ${i}: x1=${parseFloat(x1).toFixed(2)} y1=${parseFloat(y1).toFixed(2)} x2=${parseFloat(x2).toFixed(2)} y2=${parseFloat(y2).toFixed(2)} score=${parseFloat(score).toFixed(3)} classId=${classId}`);
        }
    }
}
test().catch(console.error);
