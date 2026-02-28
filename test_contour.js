function getContour(mask, w, h) {
    let startX = -1, startY = -1;
    for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
            if (mask[y * w + x] === 1) {
                startX = x; startY = y; break;
            }
        }
        if (startX !== -1) break;
    }
    if (startX === -1) return { x: [], y: [] };

    const points = [];
    let x = startX, y = startY;
    let dir = 0; 
    const dx = [1, 0, -1, 0];
    const dy = [0, 1, 0, -1];
    
    let count = 0;
    while(count < 20000) {
        points.push([x, y]);
        let found = false;
        let pDir = (dir + 3) % 4; // Turn left
        for(let i=0; i<4; i++) {
            let checkDir = (pDir + i) % 4; 
            let nx = x + dx[checkDir];
            let ny = y + dy[checkDir];
            if (nx >= 0 && nx < w && ny >= 0 && ny < h && mask[ny*w + nx] === 1) {
                x = nx; y = ny; dir = checkDir; found = true; break;
            }
        }
        if (!found) break; // single pixel
        // If we circle back to start, break.
        if (x === startX && y === startY && points.length > 2) {
            points.push([x, y]);
            break;
        }
        count++;
    }

    const smoothed = [];
    const win = 3;
    for (let i = 0; i < points.length; i++) {
        let sx = 0, sy = 0;
        let c = 0;
        for (let j = -win; j <= win; j++) {
            const idx = (i + j + points.length) % points.length;
            sx += points[idx][0];
            sy += points[idx][1];
            c++;
        }
        smoothed.push([sx / c, sy / c]);
    }

    const finalPts = smoothed.filter((_, idx) => idx % 2 === 0);

    return {
        x: finalPts.map(p => p[0] / w),
        y: finalPts.map(p => p[1] / h)
    };
}
let mask = new Uint8Array(25); // 5x5
mask[6]=1; mask[7]=1; mask[8]=1;
mask[11]=1; mask[12]=1; mask[13]=1;
mask[16]=1; mask[17]=1; mask[18]=1;
mask[12]=0; // donut!
console.log(getContour(mask, 5, 5));
