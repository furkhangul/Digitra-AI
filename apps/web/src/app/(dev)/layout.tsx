import type { Metadata } from "next";
import { notFound } from "next/navigation";
import type { ReactNode } from "react";

// Pose, rig and camera debugging pages used by the scripts in /scripts.
// They exist only on the dev server; production builds answer 404.
export const metadata: Metadata = { robots: { index: false, follow: false } };

export default function DevOnlyLayout({ children }: { children: ReactNode }) {
  if (process.env.NODE_ENV === "production") notFound();
  return children;
}
