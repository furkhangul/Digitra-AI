import { cn } from "@/lib/utils";

/**
 * Digitra mark: a "D" whose bowl is a pair of sound waves radiating off the
 * stem — sign language in, speech out. Pure SVG so it stays crisp at every
 * size; the gradient ids are static because every instance carries an
 * identical definition, so duplicate ids resolve to the same look.
 */
export function LogoMark({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 32 32" fill="none" className={cn("shrink-0", className)} aria-hidden="true">
      <defs>
        <linearGradient id="digitra-mark-grad" x1="0" y1="0" x2="32" y2="32" gradientUnits="userSpaceOnUse">
          <stop stopColor="#7b66ff" />
          <stop offset="1" stopColor="#4a2cd4" />
        </linearGradient>
      </defs>
      <rect width="32" height="32" rx="9" fill="url(#digitra-mark-grad)" />
      <rect x="7" y="6" width="3.6" height="20" rx="1.8" fill="#fff" />
      <path d="M15 10.5a6.5 6.5 0 0 1 0 11" stroke="#fff" strokeWidth="3" strokeLinecap="round" />
      <path
        d="M19.5 6.5a11 11 0 0 1 0 19"
        stroke="#fff"
        strokeOpacity="0.5"
        strokeWidth="3"
        strokeLinecap="round"
      />
    </svg>
  );
}
