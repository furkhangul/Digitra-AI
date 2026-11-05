// A tiny window-event bus that lets the voice assistant drive the page
// sections (camera, text buffer, fingerspelling player, quiz) without prop
// drilling, and lets those sections report state back.

export type AssistantCommand =
  | { type: "camera"; on: boolean }
  | { type: "text"; op: "clear" | "backspace" | "space" }
  | { type: "spell"; text: string }
  | { type: "quiz" };

const COMMAND_EVENT = "digitra:command";
const RECOGNITION_EVENT = "digitra:recognition";

export type RecognitionSnapshot = { text: string; prediction: string | null; model: string };

let lastRecognition: RecognitionSnapshot | null = null;

export function sendCommand(command: AssistantCommand) {
  window.dispatchEvent(new CustomEvent<AssistantCommand>(COMMAND_EVENT, { detail: command }));
}

export function onCommand(handler: (command: AssistantCommand) => void) {
  const listener = (event: Event) => handler((event as CustomEvent<AssistantCommand>).detail);
  window.addEventListener(COMMAND_EVENT, listener);
  return () => window.removeEventListener(COMMAND_EVENT, listener);
}

export function publishRecognition(snapshot: RecognitionSnapshot) {
  lastRecognition = snapshot;
  window.dispatchEvent(new CustomEvent<RecognitionSnapshot>(RECOGNITION_EVENT, { detail: snapshot }));
}

export function getLastRecognition() {
  return lastRecognition;
}
