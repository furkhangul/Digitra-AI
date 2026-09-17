"use client";

import { Monitor, Moon, Sun } from "lucide-react";
import { useTheme } from "./theme-provider";
import { cn } from "@/lib/utils";

const options = [
  { value: "light" as const, icon: Sun, label: "Açık tema" },
  { value: "system" as const, icon: Monitor, label: "Sistem teması" },
  { value: "dark" as const, icon: Moon, label: "Koyu tema" },
];

export function ThemeToggle() {
  const { theme, setTheme } = useTheme();

  return (
    <div
      role="radiogroup"
      aria-label="Tema seçimi"
      className="glass flex items-center gap-0.5 rounded-full p-1"
    >
      {options.map(({ value, icon: Icon, label }) => (
        <button
          key={value}
          role="radio"
          aria-checked={theme === value}
          aria-label={label}
          title={label}
          onClick={() => setTheme(value)}
          className={cn(
            "relative flex h-7 w-7 items-center justify-center rounded-full transition-colors duration-300",
            theme === value
              ? "bg-primary text-white"
              : "text-muted hover:text-foreground"
          )}
        >
          <Icon size={14} strokeWidth={2.2} />
        </button>
      ))}
    </div>
  );
}
