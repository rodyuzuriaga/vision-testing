import { InferenceSession } from 'onnxruntime-node';

async function inspect() {
    try {
        const session = await InferenceSession.create('./public/yolo26s-seg.onnx');
        console.log('Inputs:', session.inputNames);
        console.log('Outputs:', session.outputNames);
        session.outputNames.forEach(name => {
            // @ts-ignore
            console.log(`Output "${name}" shape:`, session.outputs[name]?.dims);
        });
    } catch (e) {
        console.error(e);
    }
}

inspect();
