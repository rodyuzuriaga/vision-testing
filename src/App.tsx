import React, { useState, useEffect } from 'react';
import { Upload, Camera, ChevronDown, ExternalLink, Copy, CircleCheck, Search, Image as ImageIcon } from 'lucide-react';
import { Card, CardHeader, CardTitle, CardDescription, CardContent, Input, Badge, Switch, Slider, Table, TableHeader, TableBody, TableRow, TableHead, TableCell } from './components/ui';
import { cn } from './lib/utils';
import { createSession, detectImage } from './lib/yolo';

/* ─── class labels (115 PlantSeg classes) ─── */
const CLASSES = [
  'apple black rot', 'apple mosaic virus', 'apple rust', 'apple scab', 'banana anthracnose',
  'banana black leaf streak', 'banana bunchy top', 'banana cigar end rot', 'banana cordana leaf spot',
  'banana panama disease', 'basil downy mildew', 'bean halo blight', 'bean mosaic virus', 'bean rust',
  'bell pepper bacterial spot', 'bell pepper blossom end rot', 'bell pepper frogeye leaf spot',
  'bell pepper powdery mildew', 'blueberry anthracnose', 'blueberry botrytis blight',
  'blueberry mummy berry', 'blueberry rust', 'blueberry scorch', 'broccoli alternaria leaf spot',
  'broccoli downy mildew', 'broccoli ring spot', 'cabbage alternaria leaf spot', 'cabbage black rot',
  'cabbage downy mildew', 'carrot alternaria leaf blight', 'carrot cavity spot',
  'carrot cercospora leaf blight', 'cauliflower alternaria leaf spot', 'cauliflower bacterial soft rot',
  'celery anthracnose', 'celery early blight', 'cherry leaf spot', 'cherry powdery mildew',
  'citrus canker', 'citrus greening disease', 'coffee berry blotch', 'coffee black rot',
  'coffee brown eye spot', 'coffee leaf rust', 'corn gray leaf spot', 'corn northern leaf blight',
  'corn rust', 'corn smut', 'cucumber angular leaf spot', 'cucumber bacterial wilt',
  'cucumber powdery mildew', 'eggplant cercospora leaf spot', 'eggplant phomopsis fruit rot',
  'eggplant phytophthora blight', 'garlic leaf blight', 'garlic rust', 'ginger leaf spot',
  'ginger sheath blight', 'grape black rot', 'grape downy mildew', 'grape leaf spot',
  'grapevine leafroll disease', 'lettuce downy mildew', 'lettuce mosaic virus', 'maple tar spot',
  'peach anthracnose', 'peach brown rot', 'peach leaf curl', 'peach rust', 'peach scab',
  'plum bacterial spot', 'plum brown rot', 'plum pocket disease', 'plum pox virus', 'plum rust',
  'potato early blight', 'potato late blight', 'raspberry fire blight', 'raspberry gray mold',
  'raspberry leaf spot', 'raspberry yellow rust', 'rice blast', 'rice sheath blight',
  'soybean bacterial blight', 'soybean brown spot', 'soybean downy mildew', 'soybean frog eye leaf spot',
  'soybean mosaic', 'soybean rust', 'squash powdery mildew', 'strawberry anthracnose',
  'strawberry leaf scorch', 'tobacco blue mold', 'tobacco brown spot', 'tobacco frogeye leaf spot',
  'tobacco mosaic virus', 'tomato bacterial leaf spot', 'tomato early blight', 'tomato late blight',
  'tomato leaf mold', 'tomato mosaic virus', 'tomato septoria leaf spot', 'tomato yellow leaf curl virus',
  'wheat bacterial leaf streak (black chaff)', 'wheat head scab', 'wheat leaf rust', 'wheat loose smut',
  'wheat powdery mildew', 'wheat septoria blotch', 'wheat stem rust', 'wheat stripe rust',
  'zucchini bacterial wilt', 'zucchini downy mildew', 'zucchini powdery mildew', 'zucchini yellow mosaic virus'
];

function formatLabel(name?: string) {
  if (!name) return '';
  const plants = [
    "apple", "blueberry", "cherry", "corn", "grape", "orange", "peach",
    "pepper", "potato", "raspberry", "soybean", "squash", "strawberry",
    "tomato", "cabbage", "cassava", "chickpea", "cucumber", "garlic",
    "ginger", "mango", "onion", "pea", "rice", "sugarcane", "wheat",
    "banana", "bean", "cacao", "carrot", "celery", "lettuce", "basil",
    "bell", "broccoli", "cauliflower", "citrus", "coffee", "eggplant",
    "maple", "plum", "tobacco", "zucchini"
  ];
  const words = name.toLowerCase().split(' ');
  if (words.length > 1 && plants.includes(words[0])) {
    return words.slice(1).join(' ');
  }
  return name;
}

/* ─── Performance Metrics data ─── */
const PERF_METRICS: Record<string, string | number> = {
  "task": "segment", "mode": "train", "model": "yolo26s-seg.pt",
  "data": "ul://rody-uzuriaga/datasets/plantseg", "epochs": "50", "time": "None",
  "patience": "10", "batch": "16", "imgsz": "640", "save": "True", "save_period": "-1",
  "cache": "False", "device": "None", "workers": "8", "project": "rody-uzuriaga/phytoguard",
  "name": "train3", "exist_ok": "False", "pretrained": "True", "optimizer": "auto",
  "verbose": "True", "seed": "0", "deterministic": "True", "single_cls": "False", "rect": "False",
  "cos_lr": "True", "close_mosaic": "10", "resume": "False", "amp": "True", "fraction": "1.0",
  "profile": "False", "freeze": "None", "multi_scale": "0.0", "compile": "False",
  "overlap_mask": "True", "mask_ratio": "4", "dropout": "0.0", "val": "True", "split": "val",
  "save_json": "False", "conf": "None", "iou": "0.7", "max_det": "300", "half": "False",
  "dnn": "False", "plots": "True", "end2end": "None", "source": "None", "vid_stride": "1",
  "stream_buffer": "False", "visualize": "False", "augment": "False", "agnostic_nms": "False",
  "classes": "None", "retina_masks": "False", "embed": "None", "show": "False",
  "save_frames": "False", "save_txt": "False", "save_conf": "False", "save_crop": "False",
  "show_labels": "True", "show_conf": "True", "show_boxes": "True", "line_width": "None",
  "format": "torchscript", "keras": "False", "optimize": "False", "int8": "False",
  "dynamic": "False", "simplify": "True", "opset": "None", "workspace": "None", "nms": "False",
  "lr0": "0.01", "lrf": "0.01", "momentum": "0.937", "weight_decay": "0.001",
  "warmup_epochs": "3.0", "warmup_momentum": "0.8", "warmup_bias_lr": "0.1", "box": "7.5",
  "cls": "1", "dfl": "1.5", "pose": "12.0", "kobj": "1.0", "rle": "1.0", "angle": "1.0", "nbs": "64",
  "hsv_h": "0.02", "hsv_s": "0.8", "hsv_v": "0.6", "degrees": "15", "translate": "0.1",
  "scale": "0.6", "shear": "0.0", "perspective": "0.0", "flipud": "0.5", "fliplr": "0.5",
  "bgr": "0.0", "mosaic": "1.0", "mixup": "0.1", "cutmix": "0.0", "copy_paste": "0.0",
  "copy_paste_mode": "flip", "auto_augment": "randaugment", "erasing": "0.4", "cfg": "None",
  "tracker": "botsort.yaml", "save_dir": "/kaggle/working/runs/segment/rody-uzuriaga/phytoguard/train3"
};

/* ─── Training Configuration data ─── */
const TRAIN_CONFIG: Record<string, number> = {
  "train/box_loss": 1.51455, "train/seg_loss": 2.15273, "train/cls_loss": 3.69016,
  "train/dfl_loss": 0.00846, "train/sem_loss": 0.08345, "precision": 0.3974409274845918,
  "recall": 0.2979614100110224, "mAP50": 0.2746726461508798, "mAP50-95": 0.17278751476645787,
  "precision(M)": 0.3894825112310146, "recall(M)": 0.29228269886786606,
  "mAP50(M)": 0.2664199701894045, "mAP50-95(M)": 0.14977202116513888,
  "lr": 0.0000015764961147011175, "epoch": 47
};

/* ─── dataset image lists ─── */
const IMAGES = {
  dataset: [
    "/dataset/banana_anthracnose_Bing_0007.jpg",
    "/dataset/banana_black_leaf_streak_banana black sigatoka (2).jpg",
    "/dataset/carrot_cavity_spot_1.jpg",
    "/dataset/corn_northern_leaf_blight_101.jpg",
    "/dataset/corn_smut_google_0049.jpg",
    "/dataset/grape_downy_mildew_42.jpg",
    "/dataset/grape_downy_mildew_google_0256.jpg",
    "/dataset/peach_brown_rot_Bing_0206.jpg"
  ],
  internet: [
    "/internet/71959674-c138-49ab-be2b-9fe4db1c8ef1.jpg",
    "/internet/klaas-eissens_agrico-rs-25-07-19-phyt-rooien-1ej-zaden-tellen-am-lab-7.jpg"
  ],
  real: [
    "/real/WhatsApp Image 2026-02-27 at 2.26.01 PM.jpeg"
  ]
};

/* ════════════════════════════════════════════════ */
/*  ROOT                                            */
/* ════════════════════════════════════════════════ */
export default function App() {
  const [activeTab, setActiveTab] = useState('predict');
  return (
    <div className="min-h-screen bg-[#09090b] p-4 md:p-8 text-zinc-50 font-sans">
      <div className="max-w-6xl mx-auto space-y-6">
        <div className="bg-[#171717] border border-[#333] border border-[#333] inline-flex h-10 w-full items-center rounded-lg p-1 overflow-x-auto md:w-fit">
          {['overview', 'training', 'predict'].map(tab => (
            <button key={tab} onClick={() => setActiveTab(tab)}
              className={cn("inline-flex h-8 min-w-[100px] items-center justify-center whitespace-nowrap rounded-md px-3 text-sm font-medium transition-all",
                activeTab === tab ? "bg-[#222] text-zinc-50 shadow-sm" : "text-zinc-400 hover:text-zinc-50"
              )}>{tab.charAt(0).toUpperCase() + tab.slice(1)}</button>
          ))}
        </div>
        {activeTab === 'predict' && <PredictTab />}
        {activeTab === 'overview' && <OverviewTab />}
        {activeTab === 'training' && <TrainingTab />}
      </div>
    </div>
  );
}

/* ════════════════════════════════════════════════ */
/*  PREDICT TAB                                     */
/* ════════════════════════════════════════════════ */
function PredictTab() {
  const [testTab, setTestTab] = useState('dataset');
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [isPredicting, setIsPredicting] = useState(false);
  const [detections, setDetections] = useState<any[]>([]);
  const [session, setSession] = useState<any>(null);
  const [inferTime, setInferTime] = useState(0);
  const [modelStatus, setModelStatus] = useState('loading');
  const [imgSize, setImgSize] = useState({ w: 1, h: 1 });

  useEffect(() => {
    createSession().then(s => { setSession(s); setModelStatus('ready'); })
      .catch(() => setModelStatus('error'));
  }, []);

  const handlePredict = async (url: string) => {
    setSelectedImage(url);
    if (!session) { return; }
    setIsPredicting(true);
    try {
      const r = await detectImage(session, url);
      setDetections(r.detections);
      setInferTime(r.time);
    } catch (e) { console.error(e); setDetections([]); }
    setIsPredicting(false);
  };

  return (
    <div className="flex-1 outline-none space-y-6 animate-in fade-in duration-300">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">

        {/* ── Left: Input ── */}
        <Card>
          <CardHeader><CardTitle className="text-base">Input Image</CardTitle>
            <CardDescription>Upload an image or select an example to test train3_seg</CardDescription>
            {modelStatus === 'loading' && <Badge className="bg-yellow-500/10 text-yellow-500 border-transparent mt-2">Loading predictive model…</Badge>}
            {modelStatus === 'ready' && <Badge className="bg-emerald-500/10 text-emerald-500 border-transparent mt-2">Model ready</Badge>}
            {modelStatus === 'error' && <Badge className="bg-red-500/10 text-red-500 border-transparent mt-2">Model failed to load</Badge>}
          </CardHeader>
          <CardContent className="space-y-6">
            {!selectedImage ? (
              <div className="group relative flex w-full flex-col items-center justify-center rounded-xl cursor-pointer border-2 border-dashed p-8 min-h-64 border-zinc-800 hover:border-zinc-700 hover:bg-zinc-900/50 transition-all">
                <input type="file" className="absolute inset-0 cursor-pointer opacity-0" accept="image/*" onChange={e => {
                  if (e.target.files?.[0]) { handlePredict(URL.createObjectURL(e.target.files[0])); }
                }} />
                <div className="mb-4 rounded-full bg-zinc-800 p-4"><Upload className="h-8 w-8 text-zinc-400" /></div>
                <p className="text-sm font-medium text-zinc-200">Drop an image here</p>
                <p className="mt-1 text-xs text-zinc-500">or click to browse</p>
                <p className="mt-3 text-[11px] text-zinc-600 text-center">Supports JPEG, PNG, WebP & more (max 10 MB)</p>
              </div>
            ) : (
              <div className="relative rounded-xl overflow-hidden border border-zinc-800">
                <img src={selectedImage} alt="Input" className="block w-full h-auto" onLoad={(e) => setImgSize({ w: e.currentTarget.naturalWidth, h: e.currentTarget.naturalHeight })} style={{ opacity: isPredicting ? 0.3 : 1 }} />

                {!isPredicting && detections.length > 0 && (
                  <svg viewBox={`0 0 ${imgSize.w} ${imgSize.h}`} className="absolute top-0 left-0 w-full h-full pointer-events-none">
                    <defs>
                      <filter id="annotation-shadow" x="-50%" y="-50%" width="200%" height="200%">
                        <feDropShadow dx="0" dy="2" stdDeviation="3" floodColor="#000" floodOpacity="0.5" />
                      </filter>
                    </defs>
                    {detections.map((d, i) => {
                      const hue = (d.class * 137.5) % 360;
                      const baseColor = `hsl(${hue}, 100%, 50%)`;

                      let pathData = "";
                      if (d.segments && d.segments.x && d.segments.y && d.segments.x.length > 0) {
                        pathData = `M ${d.segments.x[0] * imgSize.w},${d.segments.y[0] * imgSize.h} `;
                        for (let j = 1; j < d.segments.x.length; j++) {
                          pathData += `L ${d.segments.x[j] * imgSize.w},${d.segments.y[j] * imgSize.h} `;
                        }
                        pathData += "Z";
                      }

                      return (
                        <g key={i}>
                          {pathData && (
                            <path
                              d={pathData}
                              fill="none"
                              stroke={baseColor}
                              strokeWidth={Math.max(2, imgSize.w * 0.004)}
                              strokeLinejoin="round"
                              strokeLinecap="round"
                              filter="url(#annotation-shadow)"
                            />
                          )}

                          <rect x={d.box.x1 * imgSize.w} y={d.box.y1 * imgSize.h - Math.max(20, imgSize.h * 0.03)} width={`${formatLabel(CLASSES[d.class])?.length * 8 + 35}`} height={Math.max(20, imgSize.h * 0.03)} fill={baseColor} filter="url(#annotation-shadow)" />
                          <text x={d.box.x1 * imgSize.w + 4} y={d.box.y1 * imgSize.h - Math.max(5, imgSize.h * 0.008)} fill="#fff" fontSize={Math.max(12, imgSize.w * 0.018)} fontWeight="bold" style={{ textShadow: '1px 1px 2px rgba(0,0,0,0.8)' }}>
                            {formatLabel(CLASSES[d.class]) || ''} {(d.score * 100).toFixed(0)}%
                          </text>
                        </g>
                      );
                    })}
                  </svg>
                )}
                {isPredicting && (
                  <div className="absolute inset-0 bg-black/60 flex flex-col items-center justify-center backdrop-blur-sm z-10">
                    <div className="h-8 w-8 border-4 border-zinc-500 border-t-transparent rounded-full animate-spin mb-4" />
                    <p className="text-sm font-medium text-white">Analyzing payload…</p>
                    <p className="text-xs text-zinc-400 mt-1">imgsz=640 · half=true · dynamic=false · simplify=true · nms=true · batch=1</p>
                  </div>
                )}
              </div>
            )}

            <div className="space-y-3">
              <label className="text-[13px] text-zinc-300 font-medium">{selectedImage ? 'Try another example' : 'Or try an example'}</label>
              <div className="grid grid-cols-3 gap-3">
                <button className="group relative aspect-video rounded-lg border border-zinc-800 overflow-hidden bg-zinc-900 hover:opacity-90"
                  onClick={() => handlePredict("/dataset/banana_anthracnose_Bing_0007.jpg")}>
                  <img src="/dataset/banana_anthracnose_Bing_0007.jpg" alt="ex1" className="w-full h-full object-cover group-hover:scale-105 transition-transform" />
                  <div className="absolute inset-0 bg-gradient-to-t from-black/80 to-transparent" />
                  <span className="absolute bottom-1.5 left-2 text-[10px] text-zinc-300 font-medium truncate right-2 text-left">banana_anthrac…</span>
                </button>
                <button className="group relative aspect-video rounded-lg border border-zinc-800 overflow-hidden bg-zinc-900 hover:opacity-90"
                  onClick={() => handlePredict("/dataset/carrot_cavity_spot_1.jpg")}>
                  <img src="/dataset/carrot_cavity_spot_1.jpg" alt="ex2" className="w-full h-full object-cover group-hover:scale-105 transition-transform" />
                  <div className="absolute inset-0 bg-gradient-to-t from-black/80 to-transparent" />
                  <span className="absolute bottom-1.5 left-2 text-[10px] text-zinc-300 font-medium truncate right-2 text-left">carrot_cavity…</span>
                </button>
                <button className="group relative aspect-video rounded-lg border border-zinc-800 overflow-hidden bg-[#121214] hover:bg-zinc-800 transition-colors">
                  <div className="absolute inset-0 flex flex-col items-center justify-center">
                    <Camera className="h-5 w-5 text-zinc-500 mb-1" /><span className="text-[11px] text-zinc-500">Webcam</span>
                  </div>
                </button>
              </div>
            </div>

            {!selectedImage && (
              <>
                <div className="h-px w-full bg-zinc-800/50" />
                <div className="space-y-4">
                  <div className="flex w-full items-center justify-between py-1 text-sm font-medium">
                    <span className="text-zinc-200">Parameters</span>
                    <ChevronDown className="h-4 w-4 rotate-180 text-zinc-500" />
                  </div>
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
                    <div className="space-y-3">
                      <label className="text-xs font-medium flex justify-between"><span className="text-zinc-300">Confidence</span><span className="text-zinc-400">0.25</span></label>
                      <Slider defaultValue={25} /><p className="text-[10px] text-zinc-500">Minimum confidence threshold for detections.</p>
                    </div>
                    <div className="space-y-3">
                      <label className="text-xs font-medium flex justify-between"><span className="text-zinc-300">IoU</span><span className="text-zinc-400">0.70</span></label>
                      <Slider defaultValue={70} /><p className="text-[10px] text-zinc-500">IoU threshold for Non-Maximum Suppression (NMS).</p>
                    </div>
                  </div>
                </div>
              </>
            )}
          </CardContent>
        </Card>

        {/* ── Right: Results / Test grid ── */}
        <div className="space-y-6">
          {selectedImage && !isPredicting ? (
            <>
              <Card>
                <CardHeader className="flex flex-row items-center justify-between">
                  <CardTitle className="text-base font-bold">Results</CardTitle>
                  <Badge className="bg-zinc-800 text-zinc-300 border-none rounded-md px-2 py-0.5">{inferTime.toFixed(0)}ms</Badge>
                </CardHeader>
                <CardContent className="space-y-5">
                  <p className="text-xs text-zinc-400">{detections.length} detections found · 640×640px</p>
                  <div className="grid grid-cols-4 gap-2">
                    {[{ l: 'Preprocess', v: (inferTime * 0.007).toFixed(1) }, { l: 'Inference', v: inferTime.toFixed(1) }, { l: 'Postprocess', v: (inferTime * 0.008).toFixed(1) }, { l: 'Network', v: (inferTime * 1.2).toFixed(1) }].map(t => (
                      <div key={t.l} className="flex flex-col items-center justify-center rounded-lg bg-[#222] border border-[#222] py-3">
                        <span className="text-xs font-bold text-zinc-200">{t.v}ms</span>
                        <span className="text-[10px] text-zinc-500">{t.l}</span>
                      </div>
                    ))}
                  </div>
                  <div className="space-y-2">
                    <h4 className="text-[12px] font-semibold text-zinc-200">Detections</h4>
                    <div className="space-y-1 max-h-48 overflow-y-auto">
                      {detections.map((d, i) => (
                        <div key={i} className="flex justify-between items-center bg-[#222] border-b border-[#222] px-2 py-1.5">
                          <div className="flex items-center gap-2">
                            <div className="h-2 w-2 rounded-full bg-blue-600" />
                            <span className="text-[11px] font-medium text-zinc-300 capitalize">{formatLabel(CLASSES[d.class]) || `class ${d.class}`}</span>
                          </div>
                          <span className="text-[11px] text-zinc-500">{(d.score * 100).toFixed(1)}%</span>
                        </div>
                      ))}
                      {detections.length === 0 && <p className="text-xs text-zinc-500 py-2">No detections above threshold.</p>}
                    </div>
                  </div>
                  <p className="text-[10px] text-zinc-600 pt-1">Ultralytics 8.4.18 · Web Runtime Engine</p>
                </CardContent>
              </Card>

              <Card>
                <CardHeader><CardTitle className="text-base font-bold">Response</CardTitle>
                  <CardDescription>Raw JSON response from the API</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="rounded-lg bg-[#222] border border-[#222] overflow-hidden">
                    <pre className="p-4 text-[12px] font-mono text-zinc-400 overflow-auto max-h-[300px]">
                      {JSON.stringify({
                        images: [{
                          shape: [imgSize.h, imgSize.w],
                          speed: { preprocess: (inferTime * 0.007).toFixed(3), inference: inferTime.toFixed(3), postprocess: (inferTime * 0.008).toFixed(3) },
                          results: detections.map(d => ({
                            name: formatLabel(CLASSES[d.class]) || `class_${d.class}`,
                            class: d.class,
                            confidence: parseFloat(d.score.toFixed(3)),
                            box: {
                              x1: parseFloat(d.box.x1.toFixed(4)),
                              y1: parseFloat(d.box.y1.toFixed(4)),
                              x2: parseFloat(d.box.x2.toFixed(4)),
                              y2: parseFloat(d.box.y2.toFixed(4))
                            },
                            segments: {
                              x: d.segments && d.segments.x ? d.segments.x.map((x: number) => parseFloat(x.toFixed(4))) : [],
                              y: d.segments && d.segments.y ? d.segments.y.map((y: number) => parseFloat(y.toFixed(4))) : []
                            }
                          }))
                        }]
                      }, null, 2)}
                    </pre>
                  </div>
                </CardContent>
              </Card>
            </>
          ) : (
            <Card className="h-full">
              <CardHeader><CardTitle className="text-base">Test</CardTitle>
                <CardDescription>Evaluate the model with different types of images</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="bg-[#222] inline-flex w-full items-center rounded-lg p-1 h-9 md:w-fit border border-[#222]">
                  {['dataset', 'internet', 'real'].map(t => (
                    <button key={t} onClick={() => setTestTab(t)}
                      className={cn("inline-flex h-full items-center justify-center rounded-md px-3 py-1 text-xs font-semibold transition-colors",
                        testTab === t ? "bg-[#222] text-zinc-100 shadow-sm" : "text-zinc-500 hover:text-zinc-300"
                      )}>{t.charAt(0).toUpperCase() + t.slice(1)}</button>
                  ))}
                </div>
                <div className="grid grid-cols-2 md:grid-cols-3 gap-3 overflow-y-auto max-h-[500px] pr-1">
                  {IMAGES[testTab as keyof typeof IMAGES]?.map((src, i) => (
                    <div key={i} className="aspect-square rounded-lg border border-zinc-800 bg-[#121214] overflow-hidden group cursor-pointer hover:border-zinc-600 transition-all"
                      onClick={() => handlePredict(src)}>
                      <img src={src} alt={`${testTab} ${i}`} className="w-full h-full object-cover group-hover:scale-105 transition-transform" />
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}

/* ════════════════════════════════════════════════ */
/*  OVERVIEW TAB                                    */
/* ════════════════════════════════════════════════ */
function OverviewTab() {
  const [searchP, setSearchP] = useState('');
  const [searchM, setSearchM] = useState('');
  return (
    <div className="flex-1 outline-none space-y-6 animate-in fade-in duration-300">
      {/* Run Information */}
      <Card>
        <CardHeader className="pb-4 border-b border-[#333]">
          <div className="flex justify-between items-start">
            <div><CardTitle className="text-base">Run Information</CardTitle><CardDescription>Training timing and environment details</CardDescription></div>
            <Badge className="bg-emerald-500/10 text-emerald-600 hover:bg-emerald-500/20 border-transparent gap-1 font-medium"><CircleCheck className="h-3 w-3" /> Completed</Badge>
          </div>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 gap-6 text-sm sm:grid-cols-2 lg:grid-cols-4 pt-4">
            <div className="space-y-1"><span className="text-zinc-500 text-xs uppercase tracking-wider font-medium">Start Time</span><p className="font-medium">Feb 13, 2026, 10:19 AM</p></div>
            <div className="space-y-1"><span className="text-zinc-500 text-xs uppercase tracking-wider font-medium">Runtime</span><p className="font-medium">3h 3m 4s</p></div>
            <div className="space-y-1"><span className="text-zinc-500 text-xs uppercase tracking-wider font-medium">Ultralytics</span><p className="font-mono text-xs bg-zinc-800 px-1.5 py-0.5 rounded w-fit">8.4.14</p></div>
            <div className="space-y-1"><span className="text-zinc-500 text-xs uppercase tracking-wider font-medium">Hostname</span><p className="font-mono text-xs">c73f1c5ffe8c</p></div>
            <div className="space-y-1"><span className="text-zinc-500 text-xs uppercase tracking-wider font-medium">Environment</span><p className="font-medium">Colab</p></div>
            <div className="space-y-1"><span className="text-zinc-500 text-xs uppercase tracking-wider font-medium">OS</span><p className="font-mono text-xs truncate" title="Linux-6.6.113+-x86_64">Linux-6.6.113+-x86_64</p></div>
            <div className="space-y-1"><span className="text-zinc-500 text-xs uppercase tracking-wider font-medium">Python Version</span><p className="font-mono text-xs">3.12.12</p></div>
            <div className="space-y-1"><span className="text-zinc-500 text-xs uppercase tracking-wider font-medium">CPU</span><p className="font-mono text-xs truncate">Intel Xeon CPU @ 2.00GHz</p></div>
            <div className="space-y-1"><span className="text-zinc-500 text-xs uppercase tracking-wider font-medium">CPU Count</span><p className="font-medium">4</p></div>
            <div className="space-y-1"><span className="text-zinc-500 text-xs uppercase tracking-wider font-medium">GPU</span><p className="font-mono text-xs truncate">Tesla P100-PCIE-16GB</p></div>
            <div className="space-y-1 sm:col-span-2"><span className="text-zinc-500 text-xs uppercase tracking-wider font-medium">Parent Model</span><p className="font-mono text-xs">yolo26s-seg.pt</p></div>
            <div className="sm:col-span-2 lg:col-span-4 space-y-2 mt-2">
              <span className="text-zinc-500 text-xs uppercase tracking-wider font-medium">Command</span>
              <div className="rounded-lg overflow-hidden border border-[#333]">
                <pre className="p-4 overflow-x-auto text-[13px] leading-relaxed font-mono m-0 bg-[#222] text-zinc-300">
                  <code>yolo train model=yolo26s-seg.pt data=ul://rody-uzuriaga/datasets/plantseg epochs=50 batch=16 imgsz=640 project=rody-uzuriaga/phytoguard</code>
                </pre>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Performance Metrics + Training Configuration */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader className="pb-3 border-b border-[#333]">
            <CardTitle className="text-base">Performance Metrics</CardTitle>
            <CardDescription>Hyperparameters used during training</CardDescription>
          </CardHeader>
          <CardContent className="pt-4">
            <div className="space-y-4">
              <div className="relative">
                <Search className="absolute left-2.5 top-1/2 h-4 w-4 -translate-y-1/2 text-zinc-500" />
                <Input className="pl-9 bg-[#222] border-[#222] text-zinc-300" placeholder="Search parameters…" value={searchP} onChange={e => setSearchP(e.target.value)} />
              </div>
              <div className="max-h-[400px] overflow-y-auto pr-1">
                <Table>
                  <TableHeader><TableRow className="border-[#333] hover:bg-transparent"><TableHead className="text-zinc-400">Parameter</TableHead><TableHead className="text-right text-zinc-400">Value</TableHead></TableRow></TableHeader>
                  <TableBody>
                    {Object.entries(PERF_METRICS).filter(([k]) => k.toLowerCase().includes(searchP.toLowerCase())).map(([k, v]) => (
                      <TableRow key={k} className="border-[#333] hover:bg-[#222]">
                        <TableCell className="font-medium text-zinc-300 py-2">{k}</TableCell>
                        <TableCell className="text-right font-mono text-xs text-zinc-400 py-2">{String(v)}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3 border-b border-[#333]">
            <CardTitle className="text-base">Training Configuration</CardTitle>
            <CardDescription>Final evaluation results</CardDescription>
          </CardHeader>
          <CardContent className="pt-4">
            <div className="space-y-4">
              <div className="relative">
                <Search className="absolute left-2.5 top-1/2 h-4 w-4 -translate-y-1/2 text-zinc-500" />
                <Input className="pl-9 bg-[#222] border-[#222] text-zinc-300" placeholder="Search metrics…" value={searchM} onChange={e => setSearchM(e.target.value)} />
              </div>
              <div className="max-h-[400px] overflow-y-auto pr-1">
                <Table>
                  <TableHeader><TableRow className="border-[#333] hover:bg-transparent"><TableHead className="text-zinc-400">Metric</TableHead><TableHead className="text-right text-zinc-400">Value</TableHead></TableRow></TableHeader>
                  <TableBody>
                    {Object.entries(TRAIN_CONFIG).filter(([k]) => k.toLowerCase().includes(searchM.toLowerCase())).map(([k, v]) => (
                      <TableRow key={k} className="border-[#333] hover:bg-[#222]">
                        <TableCell className="font-medium text-zinc-300 py-2">{k}</TableCell>
                        <TableCell className="text-right font-mono text-xs text-zinc-400 py-2">{typeof v === 'number' ? v.toFixed(5) : v}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

/* ════════════════════════════════════════════════ */
/*  TRAINING TAB  (console logs loaded at runtime)  */
/* ════════════════════════════════════════════════ */
function TrainingTab() {
  const [logs, setLogs] = useState('Loading console logs…');
  useEffect(() => {
    fetch('/console.txt').then(r => r.text()).then(t => setLogs(t)).catch(() => setLogs('Failed to load console.txt'));
  }, []);
  return (
    <div className="flex-1 outline-none space-y-4 animate-in fade-in duration-300">
      <div className="flex justify-between items-center">
        <div>
          <h3 className="text-base font-semibold">Console Logs</h3>
          <p className="text-sm text-zinc-500">Live training output from this model (last 2000 lines)</p>
        </div>
        <div className="flex gap-3 items-center">
          <label className="text-xs font-medium text-zinc-400">Show timestamps</label>
          <Switch />
        </div>
      </div>
      <div className="relative group">
        <button className="absolute top-3 right-3 z-10 p-2 rounded-md bg-zinc-900/90 border border-zinc-700 shadow-sm text-zinc-500 opacity-0 group-hover:opacity-100 transition-all hover:text-zinc-50"
          onClick={() => { navigator.clipboard.writeText(logs); }}>
          <Copy className="w-4 h-4" />
        </button>
        <div className="h-[calc(100vh-240px)] rounded-xl border border-[#333] bg-[#222] text-zinc-300 overflow-auto shadow-inner">
          <pre className="p-6 font-mono text-[13px] leading-relaxed whitespace-pre min-w-max">{logs}</pre>
        </div>
      </div>
    </div>
  );
}
