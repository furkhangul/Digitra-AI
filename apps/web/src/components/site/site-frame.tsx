"use client";

import { usePathname } from "next/navigation";
import type { ReactNode } from "react";
import { Nav } from "./nav";
import { Footer } from "./footer";
import { ScrollProgress } from "./scroll-progress";

export function SiteFrame({ children }: { children: ReactNode }) {
  const pathname = usePathname();
  if (pathname === "/egitim/canli") return children;
  return <><ScrollProgress /><Nav /><main className="flex-1">{children}</main><Footer /></>;
}
