function getContourOriginal(mask, w, h) {
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
    let dir = 0; // East
    const dx = [1, 0, -1, 0];
    const dy = [0, 1, 0, -1];
    
    // Check if it's strictly a 1-pixel blob to prevent loop lock
    if (mask[startY * w + startX + 1] !== 1 && mask[(startY + 1) * w + startX] !== 1) {
        return { x: [startX / w], y: [startY / h] };
    }

    let count = 0;
    let firstDir = -1;
    while (count < 20000) {
        points.push([x, y]);
        let found = false;
        let pDir = (dir + 3) % 4; // Turn left
        for (let i = 0; i < 4; i++) {
            let checkDir = (pDir + i) % 4;
            let nx = x + dx[checkDir];
            let ny = y + dy[checkDir];
            if (nx >= 0 && nx < w && ny >= 0 && ny < h && mask[ny * w + nx] === 1) {
                x = nx; 
                y = ny; 
                dir = checkDir; 
                found = true; 
                if (firstDir === -1) firstDir = dir;
                break;
            }
        }
        if (!found) break; // single pixel
        // If we circle back to start, and we are moving in the initial starting direction
        if (x === startX && y === startY && dir === firstDir && points.length > 2) {
            points.push([x, y]);
            break;
        }
        count++;
    }

    return points;
}

const w = 10;
const h = 10;
const mask = new Uint8Array(w * h);
// create a 3x3 square
for(let y=3; y<=5; y++) {
    for(let x=3; x<=5; x++) {
        mask[y*w+x] = 1;
    }
}
// hollow out the middle? No, solid.
let pts = getContourOriginal(mask, w, h);
console.log(pts);
