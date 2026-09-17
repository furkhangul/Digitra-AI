# Digitra Landmark Personal v1.0.0

## Intended use

- Fast, local landmark inference for the Digitra live camera panel.
- One-hand `0-9` and `A-Z` technical ASL fingerspelling recognition.
- Two-hand temporal recognition for the TİD-specific characters `Ç, Ğ, İ, Ö, Ş, Ü`.

## Input and privacy

The browser extracts up to two MediaPipe hands. Only 21×3 normalized image
landmarks, optional world landmarks and handedness metadata are sent to the local
API. Raw images and video are neither transmitted nor stored.

## Limitations

- The one-hand model is ASL-based and must not be described as a complete TİD model.
- The personal adaptation was measured on one user's same-session chronological
  holdout, so its high score is not a cross-person generalization result.
- The TİD-special temporal model contains 30 sequences per class from one user and
  one session. Its 100% chronological holdout result is not cross-person accuracy.
- `J` uses a pinky-motion rule layered over the static model. A dedicated `Z`
  trajectory model is not enabled because the previous Z capture used the wrong
  finger. Static Z can still appear as a base classifier candidate.
- Confidence is a model score, not a guarantee of correctness. Low-confidence or
  unstable outputs are rejected.

## Runtime behavior

- Eight-frame smoothing for one-hand output.
- Confidence, class-margin and vote-ratio gates.
- Thirty-frame temporal window for two-hand TİD-special output.
- Hand-release gate prevents a held pose from repeatedly appending characters.
- Left/right orientation is evaluated symmetrically.
