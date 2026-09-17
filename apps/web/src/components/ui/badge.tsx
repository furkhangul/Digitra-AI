import { cn } from "@/lib/utils";
import type { ReactNode } from "react";

export function Badge({
  children,
  className,
  variant = "default",
}: {
  children: ReactNode;
  className?: string;
  variant?: "default" | "outline" | "demo";
}) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-xs font-medium tracking-wide",
        variant === "default" && "bg-primary/10 text-primary",
        variant === "outline" && "border border-border text-muted",
        variant === "demo" && "border border-warning/40 bg-warning/10 text-warning",
        className
      )}
    >
      {children}
    </span>
  );
}
