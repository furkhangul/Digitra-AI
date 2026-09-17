"use client";

import type { HandLandmarker, HandLandmarkerResult } from "@mediapipe/tasks-vision";
import {
  Camera,
  CircleStop,
  Delete,
  Eraser,
  Hand,
  LoaderCircle,
  Plus,
  Space,
  Volume2,
  Wifi,
  WifiOff,
} from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { cn } from "@/lib/utils";
import { HAND_DETECTION_OPTIONS, StableHandDetections } from "@/lib/hand-detection";
import { HandCropTracker } from "@/lib/hand-crop";
import cameraStyles from "./training-camera.module.css";

const CONNECTIONS: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [0, 9], [9, 10], [10, 11], [11, 12],
  [0, 13], [13, 14], [14, 15], [15, 16],
  [0, 17], [17, 18], [18, 19], [19, 20],
  [5, 9], [9, 13], [13, 17],
];

type Candidate = { label: string; confidence: number };

type RecognitionResult = {
  model_ready: boolean;
  model?: string;
  version?: string;
  scope?: "tid_alphabet_image" | "asl_one_hand" | "tid_specials_two_hand" | "tid_motion_experimental";
  mode?: "waiting" | "tid_image" | "one_hand" | "two_hands" | "waiting_two_hands" | "tid_motion";
  recognition_mode?: "alphabet" | "motion" | "auto";
  motion_supported?: boolean;
  motion_status?: string | null;
  confidence_kind?: "model" | "trajectory_rule" | "pose_rule";
  source?: string;
  hands_detected?: number;
  prediction?: string | null;
  candidate?: string;
  confidence?: number;
  margin?: number;
  ready?: boolean;
  frames?: number;
  candidates?: Candidate[];
  text?: string;
  added?: string | null;
  locked?: boolean;
  progress?: number;
  reject_reason?: string | null;
  latency_ms?: number;
  error?: string;
};

type CameraState = "idle" | "loading" | "running" | "error";
type ConnectionState = "idle" | "connecting" | "open" | "closed";

const HAND_CROP_SIZE = 320;
const MODEL_SEND_INTERVAL_MS = 110;
// The camera feed stays untouched for MediaPipe and the model. This filter is
// display-only so the recognizer keeps receiving the original camera pixels.
// Snapchat-style monochrome: remove all color, crush mid-tones, and keep the
// camera background close to black while the colored landmark overlay stays
// on its separate canvas.
const CAMERA_DISPLAY_FILTER = "grayscale(1) saturate(0) contrast(2.2) brightness(0.5)";

const EMPTY_RESULT: RecognitionResult = {
  model_ready: true,
  mode: "waiting",
  hands_detected: 0,
  prediction: null,
  candidate: "—",
  confidence: 0,
  ready: false,
  frames: 0,
  candidates: [],
  text: "",
  progress: 0,
};

function websocketUrl(apiBase?: string) {
  const base = apiBase ?? process.env.NEXT_PUBLIC_DIGITRA_API_URL ?? "http://127.0.0.1:8001";
  const url = new URL("/ws/recognize", base);
  url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
  return url.toString();
}

function modeLabel(result: RecognitionResult) {
  if (result.mode === "tid_motion") return "Hareketli harfler · deneme";
  if (result.mode === "tid_image") return result.motion_supported && result.recognition_mode === "auto" ? "Otomatik · sabit + hareket" : result.motion_supported ? "Sabit harfler" : "TİD alfabesi · 29 harf";
  if (result.mode === "two_hands" || result.mode === "waiting_two_hands") {
    return "TİD özel · iki el";
  }
  if (result.mode === "one_hand") return "ASL teknik · tek el";
  return "El bekleniyor";
}

export function LiveRecognition({
  apiBase,
  modelName = "TİD Robust V5",
  onConfirmedLetter,
  onRunningChange,
  expectedLetter,
  variant = "default",
}: {
  apiBase?: string;
  modelName?: string;
  onConfirmedLetter?: (letter: string) => void;
  onRunningChange?: (running: boolean) => void;
  expectedLetter?: string;
  variant?: "default" | "training";
} = {}) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const cropCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const handCropRef = useRef(new HandCropTracker());
  const detectorRef = useRef<HandLandmarker | null>(null);
  const handDetectionsRef = useRef(new StableHandDetections());
  const socketRef = useRef<WebSocket | null>(null);
  const reconnectTimerRef = useRef<number | null>(null);
  const mountedRef = useRef(true);
  const streamRef = useRef<MediaStream | null>(null);
  const animationRef = useRef<number | null>(null);
  const runningRef = useRef(false);
  const lastVideoTimeRef = useRef(-1);
  const lastSentRef = useRef(0);
  const predictionPendingRef = useRef(false);
  const lastReportedLetterRef = useRef<string | null>(null);
  const fpsWindowRef = useRef({ started: 0, frames: 0 });
  const recognitionModeRef = useRef<"alphabet" | "motion" | "auto">("auto");
  const motionFramesRef = useRef<{ timestamp_ms: number; hands: ReturnType<typeof serializeHands> }[]>([]);

  const [cameraState, setCameraState] = useState<CameraState>("idle");
  const [connectionState, setConnectionState] = useState<ConnectionState>("idle");
  const [result, setResult] = useState<RecognitionResult>(EMPTY_RESULT);
  const [fps, setFps] = useState<number | null>(null);
  const [error, setError] = useState("");

  const clearCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    const context = canvas?.getContext("2d");
    if (canvas && context) context.clearRect(0, 0, canvas.width, canvas.height);
  }, []);

  const stop = useCallback(() => {
    runningRef.current = false;
    if (animationRef.current !== null) cancelAnimationFrame(animationRef.current);
    animationRef.current = null;
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    predictionPendingRef.current = false;
    lastReportedLetterRef.current = null;
    handDetectionsRef.current.reset();
    handCropRef.current.reset();
    motionFramesRef.current = [];
    if (socketRef.current?.readyState === WebSocket.OPEN) {
      socketRef.current.send(JSON.stringify({ hands: [], motion_frames: [] }));
    }
    setCameraState("idle");
    onRunningChange?.(false);
    setFps(null);
    clearCanvas();
  }, [clearCanvas, onRunningChange]);

  useEffect(() => {
    return () => {
      mountedRef.current = false;
      stop();
      socketRef.current?.close();
      socketRef.current = null;
      if (reconnectTimerRef.current !== null) {
        window.clearTimeout(reconnectTimerRef.current);
      }
      reconnectTimerRef.current = null;
      detectorRef.current?.close();
      detectorRef.current = null;
    };
  }, [stop]);

  const ensureDetector = useCallback(async () => {
    if (detectorRef.current) return detectorRef.current;
    const { FilesetResolver, HandLandmarker: HandLandmarkerClass } = await import(
      "@mediapipe/tasks-vision"
    );
    const vision = await FilesetResolver.forVisionTasks("/mediapipe/wasm");
    const options = {
      baseOptions: { modelAssetPath: "/mediapipe/hand_landmarker.task", delegate: "GPU" as const },
      runningMode: "VIDEO" as const,
      numHands: 2,
      ...HAND_DETECTION_OPTIONS,
    };
    try {
      detectorRef.current = await HandLandmarkerClass.createFromOptions(vision, options);
    } catch {
      detectorRef.current = await HandLandmarkerClass.createFromOptions(vision, {
        ...options,
        baseOptions: { modelAssetPath: "/mediapipe/hand_landmarker.task", delegate: "CPU" },
      });
    }
    return detectorRef.current;
  }, []);

  const drawLandmarks = useCallback((output: HandLandmarkerResult) => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const context = canvas?.getContext("2d");
    if (!video || !canvas || !context || !video.videoWidth || !video.videoHeight) return;

    if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
    }
    context.clearRect(0, 0, canvas.width, canvas.height);

    output.landmarks.slice(0, 2).forEach((landmarks, handIndex) => {
      // Keep the hand skeleton colored so it remains easy to follow over the
      // deliberately dark, monochrome camera treatment.
      const color = handIndex === 0 ? "#22d3ee" : "#a78bfa";
      const xs = landmarks.map((point) => point.x * canvas.width);
      const ys = landmarks.map((point) => point.y * canvas.height);
      const padding = 14;

      context.strokeStyle = `${color}99`;
      context.lineWidth = Math.max(1.5, canvas.width / 420);
      context.setLineDash([8, 7]);
      context.strokeRect(
        Math.min(...xs) - padding,
        Math.min(...ys) - padding,
        Math.max(...xs) - Math.min(...xs) + padding * 2,
        Math.max(...ys) - Math.min(...ys) + padding * 2,
      );
      context.setLineDash([]);
      context.strokeStyle = color;
      context.lineCap = "round";
      context.lineJoin = "round";
      context.lineWidth = Math.max(2.2, canvas.width / 300);
      for (const [start, end] of CONNECTIONS) {
        context.beginPath();
        context.moveTo(xs[start], ys[start]);
        context.lineTo(xs[end], ys[end]);
        context.stroke();
      }
      landmarks.forEach((point, index) => {
        context.beginPath();
        context.fillStyle = index === 0 ? "#ffffff" : color;
        context.arc(
          point.x * canvas.width,
          point.y * canvas.height,
          index === 0 ? 5 : 3.5,
          0,
          Math.PI * 2,
        );
        context.fill();
      });
    });
  }, []);

  const serializeHands = useCallback((output: HandLandmarkerResult) => {
    const handedness = output.handedness ?? output.handednesses;
    return output.landmarks.slice(0, 2).map((landmarks, index) => {
      const category = handedness?.[index]?.[0];
      const worldLandmarks = output.worldLandmarks?.[index];
      return {
        // MediaPipe noktaları sınıf nesneleridir. JSON.stringify bunları nesne
        // olarak yazar; API ise her nokta için [x, y, z] sayısal dizisi bekler.
        landmarks: landmarks.map((point) => [point.x, point.y, point.z]),
        world_landmarks:
          worldLandmarks?.map((point) => [point.x, point.y, point.z]) ?? null,
        handedness: category?.categoryName ?? "Unknown",
        handedness_score: category?.score ?? 0,
      };
    });
  }, []);

  const serializeHandCrop = useCallback(
    (video: HTMLVideoElement, output: HandLandmarkerResult) => {
      const crop = handCropRef.current.update(
        output.landmarks, video.videoWidth, video.videoHeight, performance.now(),
      );
      if (!crop) {
        const canvas = cropCanvasRef.current;
        canvas?.getContext("2d")?.clearRect(0, 0, canvas.width, canvas.height);
        return null;
      }
      const sourceWidth = crop.width;
      const sourceHeight = crop.height;

      const cropCanvas = cropCanvasRef.current ?? document.createElement("canvas");
      cropCanvasRef.current = cropCanvas;
      cropCanvas.width = HAND_CROP_SIZE;
      cropCanvas.height = HAND_CROP_SIZE;
      const context = cropCanvas.getContext("2d");
      if (!context) return null;
      // Keep the crop payload stable for the model. The black-and-white look
      // is applied with CSS to the on-screen canvas below.
      context.fillStyle = "rgb(114, 114, 114)";
      context.fillRect(0, 0, HAND_CROP_SIZE, HAND_CROP_SIZE);
      context.imageSmoothingEnabled = true;
      context.imageSmoothingQuality = "high";
      const scale = Math.min(
        HAND_CROP_SIZE / sourceWidth,
        HAND_CROP_SIZE / sourceHeight,
      );
      const targetWidth = sourceWidth * scale;
      const targetHeight = sourceHeight * scale;
      context.drawImage(
        video,
        crop.x,
        crop.y,
        sourceWidth,
        sourceHeight,
        (HAND_CROP_SIZE - targetWidth) / 2,
        (HAND_CROP_SIZE - targetHeight) / 2,
        targetWidth,
        targetHeight,
      );
      return cropCanvas.toDataURL("image/jpeg", 0.92);
    },
    [],
  );

  const connectSocket = useCallback(function connect() {
    if (
      socketRef.current?.readyState === WebSocket.OPEN ||
      socketRef.current?.readyState === WebSocket.CONNECTING
    ) {
      return;
    }
    if (reconnectTimerRef.current !== null) {
      window.clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }
    setConnectionState("connecting");
    const socket = new WebSocket(websocketUrl(apiBase));
    socketRef.current = socket;
    socket.onopen = () => {
      setConnectionState("open");
      setError("");
      motionFramesRef.current = [];
    };
    socket.onmessage = (event) => {
      predictionPendingRef.current = false;
      const payload = JSON.parse(event.data) as RecognitionResult;
      if (payload.motion_supported && payload.mode === "waiting" && payload.frames === undefined && recognitionModeRef.current !== "alphabet") {
        socket.send(JSON.stringify({ action: "set_recognition_mode", recognition_mode: recognitionModeRef.current }));
      }
      if (payload.error) setError(payload.error);
      if (payload.model_ready === false) {
        setError(payload.error ?? "Harf tanıma modeli yüklenemedi. Lütfen yeniden dene.");
      }
      setResult((current) => ({ ...current, ...payload }));
    };
    socket.onerror = () => {
      predictionPendingRef.current = false;
      setConnectionState("closed");
      setError("Tanıma servisine bağlanılamadı. Bağlantı yeniden kurulduğunda harf tanımayı deneyebilirsin.");
    };
    socket.onclose = () => {
      predictionPendingRef.current = false;
      setConnectionState("closed");
      if (mountedRef.current) {
        reconnectTimerRef.current = window.setTimeout(connect, 1500);
      }
    };
  }, [apiBase]);

  useEffect(() => {
    mountedRef.current = true;
    connectSocket();
  }, [connectSocket]);

  useEffect(() => {
    if (!onConfirmedLetter) return;
    if (result.ready && result.prediction) {
      if (lastReportedLetterRef.current !== result.prediction) {
        lastReportedLetterRef.current = result.prediction;
        onConfirmedLetter(result.prediction);
      }
    } else if (!result.ready) {
      // Releasing the hand arms the same letter again. This is needed for
      // repeated letters such as the two N's in “ANNE”.
      lastReportedLetterRef.current = null;
    }
  }, [onConfirmedLetter, result.prediction, result.ready]);

  const frameLoop = useCallback(function runFrame() {
    if (!runningRef.current) return;
    const video = videoRef.current;
    const detector = detectorRef.current;
    if (video && detector && video.readyState >= 2 && video.currentTime !== lastVideoTimeRef.current) {
      lastVideoTimeRef.current = video.currentTime;
      const now = performance.now();
      const output = handDetectionsRef.current.update(
        detector.detectForVideo(video, now),
        now,
        video.videoWidth / video.videoHeight,
      );
      drawLandmarks(output);
      const hands = serializeHands(output);
      if (recognitionModeRef.current !== "alphabet") {
        motionFramesRef.current.push({ timestamp_ms: now, hands });
        if (motionFramesRef.current.length > 60) motionFramesRef.current.shift();
      }

      const fpsWindow = fpsWindowRef.current;
      if (fpsWindow.started === 0) fpsWindow.started = now;
      fpsWindow.frames += 1;
      if (now - fpsWindow.started >= 1000) {
        setFps(Math.round((fpsWindow.frames * 1000) / (now - fpsWindow.started)));
        fpsWindow.started = now;
        fpsWindow.frames = 0;
      }

      if (
        now - lastSentRef.current >= MODEL_SEND_INTERVAL_MS &&
        !predictionPendingRef.current &&
        socketRef.current?.readyState === WebSocket.OPEN
      ) {
        const image = serializeHandCrop(video, output);
        const motion_frames = motionFramesRef.current;
        motionFramesRef.current = [];
        socketRef.current.send(JSON.stringify({ hands, image, motion_frames, aspect_ratio: video.videoWidth / video.videoHeight }));
        predictionPendingRef.current = true;
        lastSentRef.current = now;
      }
    }
    animationRef.current = requestAnimationFrame(runFrame);
  }, [drawLandmarks, serializeHandCrop, serializeHands]);

  const start = useCallback(async () => {
    setCameraState("loading");
    setError("");
    try {
      await ensureDetector();
      if (!mountedRef.current) return;
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      });
      if (!mountedRef.current) {
        stream.getTracks().forEach((track) => track.stop());
        return;
      }
      streamRef.current = stream;
      if (!videoRef.current) throw new Error("Video alanı hazırlanamadı.");
      videoRef.current.srcObject = stream;
      await videoRef.current.play();
      runningRef.current = true;
      handDetectionsRef.current.reset();
      handCropRef.current = new HandCropTracker();
      motionFramesRef.current = [];
      lastVideoTimeRef.current = -1;
      predictionPendingRef.current = false;
      fpsWindowRef.current = { started: 0, frames: 0 };
      setCameraState("running");
      onRunningChange?.(true);
      connectSocket();
      animationRef.current = requestAnimationFrame(frameLoop);
    } catch (cause) {
      const message = cause instanceof Error ? cause.message : "Bilinmeyen kamera hatası";
      setError(`Kamera başlatılamadı: ${message}`);
      setCameraState("error");
      stop();
      setCameraState("error");
    }
  }, [connectSocket, ensureDetector, frameLoop, onRunningChange, stop]);

  const sendAction = useCallback((action: "append" | "space" | "backspace" | "clear") => {
    if (socketRef.current?.readyState !== WebSocket.OPEN) return;
    socketRef.current.send(JSON.stringify({ action }));
  }, []);

  const changeRecognitionMode = useCallback((mode: "alphabet" | "motion" | "auto") => {
    if (socketRef.current?.readyState !== WebSocket.OPEN) return;
    recognitionModeRef.current = mode;
    motionFramesRef.current = [];
    socketRef.current.send(JSON.stringify({ action: "set_recognition_mode", recognition_mode: mode }));
  }, []);

  const speak = useCallback(() => {
    if (!result.text || !("speechSynthesis" in window)) return;
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(result.text);
    utterance.lang = "tr-TR";
    window.speechSynthesis.speak(utterance);
  }, [result.text]);

  const confidence = result.confidence ?? 0;
  const displayedLetter = result.prediction ?? (result.candidate === "?" ? "—" : result.candidate) ?? "—";
  const running = cameraState === "running";
  const pathRule = result.confidence_kind === "trajectory_rule" || result.confidence_kind === "pose_rule";
  const motionMode = result.recognition_mode === "motion";
  const autoMode = result.recognition_mode === "auto";
  const trainingVariant = variant === "training";
  const motionHint = result.motion_status === "TWO_HANDS_REQUIRED"
    ? "Hareketi izlemek için iki el de görünmeli."
    : result.motion_status === "MOTION_REQUIRED"
      ? "El şeklini koruyarak harfin hareketini tamamla."
      : result.motion_status === "USE_MOTION_MODE"
        ? "Bu harf hareketle tamamlanır. Hareketli harfler seçeneğine geç."
        : result.ready
          ? (result.confidence_kind !== "trajectory_rule" ? "Sabit harf hazır. Yeni harf için ellerini kısa süre indir." : "Hareket tamamlandı. Yeni harf için ellerini kısa süre indir.")
          : autoMode
            ? "Sabit harfi 1–2 saniye koru; J/Ğ gibi hareketli harflerde iki elini gösterip hareketi tamamla."
            : "İki elini göster; hareketi 3–4 saniye boyunca yap, ardından son konumda kısa süre bekle.";

  if (trainingVariant) {
    const targetMatches = running && result.candidate === expectedLetter;
    const targetProgress = targetMatches ? (result.ready ? 100 : Math.round((result.progress ?? 0) * 100)) : 0;
    const dynamicTarget = !!expectedLetter && "ÇĞİJÖŞÜ".includes(expectedLetter);
    const trainingHint = !result.hands_detected ? "El görünmüyor; ellerini kadraja getir."
      : dynamicTarget && result.hands_detected < 2 ? "Hareket için iki elin de görünmesi gerekiyor."
      : result.ready && targetMatches ? "Harf eşleşti"
      : expectedLetter === "Ğ" ? "G şeklini koru; işaret parmağını birkaç kez yukarı-aşağı oynat."
      : dynamicTarget ? "Hareketi tamamla; ardından kısa süre bekle."
      : targetMatches ? "El şeklini kısa süre koru."
      : expectedLetter === "N" ? "Bir elde iki, diğerinde tek parmak aşağı; uçları yaklaştır."
      : "Modeldeki harfi göster.";
    return (
      <div className={cameraStyles.camera}>
        <video ref={videoRef} autoPlay playsInline muted className={cameraStyles.video}
          style={{ transform: "scaleX(-1)", filter: CAMERA_DISPLAY_FILTER }} aria-label="Aynalı canlı kamera görüntüsü" />
        <canvas ref={canvasRef} className={cameraStyles.landmarks} style={{ transform: "scaleX(-1)" }} aria-hidden="true" />
        <canvas ref={cropCanvasRef} width={HAND_CROP_SIZE} height={HAND_CROP_SIZE} hidden />
        <div className={cameraStyles.vignette} />
        <div className={cameraStyles.status}>
          <span className={connectionState === "open" ? cameraStyles.connected : cameraStyles.disconnected} />
          {connectionState !== "open" ? "Tanıma bağlantısı bekleniyor" : running ? "Kamera açık" : "Kamera hazır"}
        </div>
        {!running && <div className={cameraStyles.startArea}>
          <div className={cameraStyles.handRing}><Hand size={42} strokeWidth={1.1} /><span /></div>
          <h2>Sıra senin ellerinde.</h2>
          <p>Kameranı aç, modeldeki harfi göster.</p>
          <button type="button" onClick={start} disabled={cameraState === "loading"} className={cameraStyles.startButton}>
            {cameraState === "loading" ? <LoaderCircle size={17} className="animate-spin" /> : <Camera size={17} />}
            {cameraState === "loading" ? "Kamera hazırlanıyor…" : "Kamerayı aç"}
          </button>
        </div>}
        {running && <>
          <div className={cameraStyles.frameGuide} aria-hidden="true"><i /><i /><i /><i /></div>
          <div className={cameraStyles.handCount}>{result.hands_detected ?? 0} el görünür · {displayedLetter === "—" ? "Henüz eşleşme yok" : `${result.ready ? "Algılanan" : "Aday"}: ${displayedLetter}`}</div>
          <div className={cameraStyles.holdProgress}>
            <span>{trainingHint}</span>
            <div role="progressbar" aria-label="Harf kararlılığı" aria-valuenow={targetProgress} aria-valuemin={0} aria-valuemax={100}><i style={{ width: `${targetProgress}%` }} /></div>
          </div>
          <button type="button" onClick={stop} className={cameraStyles.stopButton} aria-label="Kamerayı durdur" title="Kamerayı durdur"><CircleStop size={20} /></button>
        </>}
        {error && <p role="alert" className={cameraStyles.error}>{error}</p>}
      </div>
    );
  }

  return (
    <div className={cn(
      "overflow-hidden bg-surface",
      trainingVariant ? "rounded-none border-0 bg-black shadow-none" : "rounded-2xl border border-border shadow-[var(--shadow-soft)]",
    )}>
      <div className={cn("flex items-center gap-2 border-b border-border bg-surface-2 px-4 py-3", trainingVariant && "hidden")}>
        <span className="h-3 w-3 rounded-full bg-[#ff5f57]" />
        <span className="h-3 w-3 rounded-full bg-[#febc2e]" />
        <span className="h-3 w-3 rounded-full bg-[#28c840]" />
        <p className="mx-auto text-xs font-medium text-muted">Digitra · Canlı harf tanıma</p>
        {connectionState === "open" ? (
          <Wifi size={14} className="text-emerald-500" aria-label="Model bağlantısı açık" />
        ) : (
          <WifiOff size={14} className="text-muted" aria-label="Model bağlantısı kapalı" />
        )}
      </div>

      {result.motion_supported && !trainingVariant && (
        <div className="border-b border-border px-4 py-3">
          <div className="flex flex-wrap gap-2" role="group" aria-label="Tanıma türü">
            {([['auto', 'Otomatik · tüm harfler'], ['alphabet', 'Sabit harfler'], ['motion', 'Hareketli harfler · deneme']] as const).map(([mode, label]) => (
              <button key={mode} type="button" aria-pressed={result.recognition_mode === mode}
                disabled={connectionState !== "open"} onClick={() => changeRecognitionMode(mode)}
                className={cn("rounded-full border px-3.5 py-2 text-xs font-medium disabled:opacity-40",
                  result.recognition_mode === mode ? "border-primary bg-primary/10 text-primary" : "border-border text-muted")}>
                {label}
              </button>
            ))}
          </div>
          {(motionMode || autoMode) && <p className="mt-2 text-xs leading-relaxed text-muted">
            Sabit harfin şeklini kısa süre koru; J ve Ğ gibi harflerde hareketi tamamla.
            Hareket takibi deneme aşamasındadır; canlı kamera başarısı henüz ölçülmedi.
          </p>}
        </div>
      )}

      <div className={cn(
        "relative overflow-hidden bg-[#070711]",
        trainingVariant ? "aspect-[4/3] min-h-[420px] sm:min-h-[540px]" : "aspect-[16/10]",
      )}>
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className="absolute inset-0 h-full w-full object-contain"
          style={{ transform: "scaleX(-1)", filter: CAMERA_DISPLAY_FILTER }}
          aria-label="Canlı kamera görüntüsü"
        />
        <canvas
          ref={canvasRef}
          className="pointer-events-none absolute inset-0 h-full w-full object-contain"
          style={{ transform: "scaleX(-1)" }}
          aria-hidden="true"
        />

        {!running && !trainingVariant && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-4 bg-[radial-gradient(circle_at_center,rgba(83,60,180,0.2),transparent_60%)] text-white">
            <span className="flex h-16 w-16 items-center justify-center rounded-2xl border border-white/10 bg-white/5">
              {cameraState === "loading" ? (
                <LoaderCircle className="animate-spin" size={27} />
              ) : (
                <Camera size={27} />
              )}
            </span>
            <div className="text-center">
              <p className="font-medium">
                {cameraState === "loading" ? "El modeli hazırlanıyor" : "Kamera kapalı"}
              </p>
              <p className="mt-1 text-xs text-white/55">Başlamak için kamera izni ver ve ellerini göster</p>
            </div>
          </div>
        )}

        <div className="absolute top-4 left-4 flex flex-wrap gap-2">
          <span className="rounded-full bg-black/55 px-3 py-1.5 text-xs text-white/85 backdrop-blur">
            {modeLabel(result)}
          </span>
          {running && (
            <span className="rounded-full bg-black/55 px-3 py-1.5 text-xs text-white/70 backdrop-blur">
              {result.hands_detected ?? 0} el · kadrajda
            </span>
          )}
        </div>

        {running && (
          <div className="absolute top-4 right-4 rounded-2xl border border-white/10 bg-black/55 px-4 py-3 text-white backdrop-blur-xl">
            <p className="text-[10px] uppercase tracking-[0.15em] text-white/55">
              {result.ready ? "Kararlı tahmin" : "Aday"}
            </p>
            <div className="mt-1 flex items-end gap-3">
              <span className={cn("text-4xl font-semibold", result.ready ? "text-white" : "text-white/60")}>
                {displayedLetter}
              </span>
              <span className="pb-1 text-xs text-white/60">{pathRule ? (result.ready ? "Eşleşti" : "Kontrol ediliyor") : `%${(confidence * 100).toFixed(1)}`}</span>
            </div>
          </div>
        )}

        {running && (
          <div className="absolute inset-x-4 bottom-4 rounded-2xl border border-white/10 bg-black/60 p-3 text-white backdrop-blur-xl">
            <div className="flex items-center justify-between gap-3 text-[11px] text-white/60">
              <span>{result.locked ? "Yeni harf için elini indir" : motionMode ? "Hareket takibi" : "Kararlılık"}</span>
              <span>{result.ready ? "Hazır" : `${result.frames ?? 0} kare`}</span>
            </div>
            <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-white/10">
              <div
                className="h-full rounded-full bg-gradient-to-r from-cyan-400 to-violet-400 transition-[width] duration-100"
                style={{ width: `${Math.round((result.progress ?? 0) * 100)}%` }}
              />
            </div>
          </div>
        )}

        {trainingVariant && !running && (
          <button type="button" onClick={start} disabled={cameraState === "loading"} className="absolute bottom-8 left-1/2 z-20 inline-flex -translate-x-1/2 items-center gap-2 rounded-full bg-white px-6 py-3 text-sm font-semibold text-black shadow-xl disabled:opacity-50">
            {cameraState === "loading" ? <LoaderCircle size={16} className="animate-spin" /> : <Camera size={16} />}
            Kamerayı başlat
          </button>
        )}
        {trainingVariant && running && (
          <button type="button" onClick={stop} className="absolute bottom-6 right-6 z-20 inline-flex items-center gap-2 rounded-full border border-white/20 bg-black/60 px-4 py-2.5 text-sm font-medium text-white backdrop-blur hover:border-white/50">
            <CircleStop size={15} /> Durdur
          </button>
        )}
        {trainingVariant && error && <p className="absolute bottom-5 left-5 right-5 z-20 rounded-xl border border-red-400/25 bg-black/75 px-3 py-2 text-xs text-red-200" role="alert">{error}</p>}
      </div>

      <div className={cn("flex items-center gap-4 border-t border-border px-4 py-3", (!running || trainingVariant) && "hidden")}>
        <canvas
          ref={cropCanvasRef}
          width={HAND_CROP_SIZE}
          height={HAND_CROP_SIZE}
          className="h-36 w-36 shrink-0 rounded-lg bg-black object-contain sm:h-40 sm:w-40"
          style={{ transform: "scaleX(-1)", filter: CAMERA_DISPLAY_FILTER }}
          aria-label="Siyah-beyaz el odaklı önizleme"
        />
        <div className="text-xs leading-relaxed text-muted">
          <p className="font-medium text-foreground">Siyah-beyaz el odaklı görünüm</p>
          <p className="mt-1">Bu filtre yalnızca önizlemeyi değiştirir. Tanıma için el bölgesinin renkli görüntüsü kullanılır.</p>
          <p className="mt-1">İki el kullanıyorsan ikisi de bu alanda görünmeli.</p>
        </div>
      </div>

      <div className={cn("grid grid-cols-3 divide-x divide-border border-y border-border text-center text-xs", trainingVariant && "hidden")}>
        <div className="px-2 py-3">
          <p className="text-muted">Model</p>
          <p className="mt-0.5 truncate font-medium">
            {pathRule ? `${displayedLetter} · ${result.confidence_kind === "pose_rule" ? "Parmak düzeni" : "Hareket kontrolü"}` : motionMode ? "TİD hareket · v1" : result.version
              ? result.model === "digitra-tid-robust"
                ? `TİD Robust v${result.version}`
                : result.model === "digitra-tid-augmented"
                  ? `TİD V6 · v${result.version}`
                : `Landmark v${result.version}`
              : "—"}
          </p>
        </div>
        <div className="px-2 py-3">
          <p className="text-muted">{pathRule ? "Geometrik kontrol" : "Güven"}</p>
          <p className="mt-0.5 font-medium">{pathRule ? (result.ready ? "Eşleşti" : "Bekleniyor") : confidence ? `%${(confidence * 100).toFixed(1)}` : "—"}</p>
        </div>
        <div className="px-2 py-3">
          <p className="text-muted">Gecikme · FPS</p>
          <p className="mt-0.5 font-medium">
            {result.latency_ms !== undefined ? `${result.latency_ms.toFixed(1)} ms` : "—"} · {fps ?? "—"}
          </p>
        </div>
      </div>

      <div className={cn("p-4 sm:p-5", trainingVariant && "hidden")}>
        <div className="flex min-h-14 items-center justify-between gap-4 rounded-xl border border-border bg-surface-2 px-4 py-3">
          <div className="min-w-0">
            <p className="text-[10px] font-medium uppercase tracking-[0.15em] text-muted">Oluşan metin</p>
            <p className="mt-1 min-h-6 truncate text-lg font-semibold" aria-live="polite">
              {result.text || "Henüz harf eklenmedi"}
            </p>
          </div>
          <button
            type="button"
            onClick={speak}
            disabled={!result.text}
            className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl border border-border text-muted transition hover:border-primary/40 hover:text-primary disabled:cursor-not-allowed disabled:opacity-35"
            aria-label="Metni seslendir"
          >
            <Volume2 size={17} />
          </button>
        </div>

        {(result.candidates?.length ?? 0) > 0 && (
          <div className="mt-3 flex items-center gap-2 overflow-x-auto text-xs">
            <span className="shrink-0 text-muted">İlk 3:</span>
            {result.candidates?.map((candidate) => (
              <span key={candidate.label} className="shrink-0 rounded-full border border-border px-2.5 py-1">
                {candidate.label} %{(candidate.confidence * 100).toFixed(0)}
              </span>
            ))}
          </div>
        )}

        <div className="mt-4 flex flex-wrap gap-2">
          {!running ? (
            <button
              type="button"
              onClick={start}
              disabled={cameraState === "loading"}
              className="inline-flex cursor-pointer items-center gap-2 rounded-full px-5 py-2.5 text-sm font-semibold text-white shadow disabled:cursor-wait disabled:opacity-60"
              style={{ background: "var(--grad-primary)" }}
            >
              {cameraState === "loading" ? <LoaderCircle size={16} className="animate-spin" /> : <Camera size={16} />}
              Kamerayı başlat
            </button>
          ) : (
            <button
              type="button"
              onClick={stop}
              className="inline-flex cursor-pointer items-center gap-2 rounded-full border border-border px-4 py-2.5 text-sm font-medium hover:border-danger/40 hover:text-danger"
            >
              <CircleStop size={16} /> Kamerayı durdur
            </button>
          )}
          <button
            type="button"
            onClick={() => sendAction("append")}
            disabled={!result.ready || connectionState !== "open"}
            className="inline-flex cursor-pointer items-center gap-1.5 rounded-full border border-border px-3.5 py-2.5 text-sm font-medium hover:border-primary/40 disabled:cursor-not-allowed disabled:opacity-35"
          >
            <Plus size={15} /> Harfi ekle
          </button>
          <button
            type="button"
            onClick={() => sendAction("space")}
            disabled={connectionState !== "open"}
            className="inline-flex cursor-pointer items-center gap-1.5 rounded-full border border-border px-3.5 py-2.5 text-sm font-medium hover:border-primary/40 disabled:opacity-35"
          >
            <Space size={15} /> Boşluk
          </button>
          <button
            type="button"
            onClick={() => sendAction("backspace")}
            disabled={connectionState !== "open"}
            className="inline-flex cursor-pointer items-center gap-1.5 rounded-full border border-border px-3.5 py-2.5 text-sm font-medium hover:border-primary/40 disabled:opacity-35"
          >
            <Delete size={15} /> Sil
          </button>
          <button
            type="button"
            onClick={() => sendAction("clear")}
            disabled={connectionState !== "open"}
            className="inline-flex cursor-pointer items-center gap-1.5 rounded-full border border-border px-3.5 py-2.5 text-sm font-medium hover:border-primary/40 disabled:opacity-35"
          >
            <Eraser size={15} /> Temizle
          </button>
        </div>

        <div className="mt-4 flex items-start gap-3 rounded-xl bg-primary/5 px-3.5 py-3 text-xs leading-relaxed text-muted">
          <Hand size={16} className="mt-0.5 shrink-0 text-primary" />
          <p>
            {motionMode || autoMode || result.motion_status === "USE_MOTION_MODE" ? motionHint :
              `${modelName} ile sabit harfleri denemek için ellerini görüntüde tutup 1–2 saniye sabitle; harf otomatik eklendikten sonra tekrar için elini kısa süre indir.`}
          </p>
        </div>

        {error && (
          <p className="mt-3 rounded-xl border border-red-500/20 bg-red-500/5 px-3.5 py-3 text-xs text-red-600 dark:text-red-300" role="alert">
            {error}
          </p>
        )}
      </div>

    </div>
  );
}
