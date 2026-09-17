"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import { Menu, X } from "lucide-react";
import { Container } from "@/components/ui/container";
import { ThemeToggle } from "./theme-toggle";
import { LogoMark } from "./logo";
import { cn } from "@/lib/utils";

const LINKS = [
  { href: "#ceviri", label: "Harf tanıma" },
  { href: "#egitim", label: "3D eğitim" },
  { href: "#how-it-works", label: "Neden Digitra" },
  { href: "#architecture", label: "Teknoloji" },
  { href: "#science", label: "Doğrulama" },
  { href: "#datasets", label: "Veri kaynakları" },
  { href: "#faq", label: "SSS" },
];

export function Nav() {
  const [scrolled, setScrolled] = useState(false);
  const [revealed, setRevealed] = useState(false);
  const [open, setOpen] = useState(false);
  const pathname = usePathname();

  useEffect(() => {
    const onScroll = () => {
      setScrolled(window.scrollY > 12);
      // The intro poster carries its own corner logo, so the bar stays out of
      // the way until the scroll-through transition has handed off.
      setRevealed(window.scrollY > window.innerHeight * 1.6);
    };
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("resize", onScroll);
    return () => {
      window.removeEventListener("scroll", onScroll);
      window.removeEventListener("resize", onScroll);
    };
  }, []);

  return (
    <header
      className={cn(
        "fixed inset-x-0 top-0 z-50 transition-all duration-500",
        revealed ? "translate-y-0 opacity-100" : "pointer-events-none -translate-y-6 opacity-0"
      )}
    >
      <Container>
        <div
          className={cn(
            "mt-4 flex items-center justify-between rounded-full px-4 py-2.5 transition-all duration-500",
            scrolled ? "glass shadow-lg" : "border border-transparent"
          )}
        >
          <Link
            href="/"
            onClick={(e) => {
              if (pathname === "/") {
                e.preventDefault();
                setOpen(false);
                window.scrollTo({ top: 0, behavior: "smooth" });
              }
            }}
            className="flex cursor-pointer items-center gap-2 pl-1 text-base font-semibold tracking-tight"
          >
            <LogoMark className="h-8 w-8" />
            Digitra
          </Link>

          <nav className="hidden items-center gap-1 lg:flex">
            {LINKS.map((link) => (
              <a
                key={link.href}
                href={link.href}
                className="rounded-full px-4 py-2 text-sm text-muted transition-colors duration-200 hover:bg-surface-2 hover:text-foreground"
              >
                {link.label}
              </a>
            ))}
          </nav>

          <div className="flex items-center gap-3">
            <div className="hidden sm:block">
              <ThemeToggle />
            </div>
            <a
              href="#ceviri"
              className="hidden rounded-full bg-foreground px-4 py-2 text-sm font-medium text-background transition-transform duration-300 hover:-translate-y-0.5 sm:inline-flex"
            >
              Hemen dene
            </a>
            <button
              onClick={() => setOpen((v) => !v)}
              className="glass flex h-9 w-9 items-center justify-center rounded-full lg:hidden"
              aria-label={open ? "Menüyü kapat" : "Menüyü aç"}
              aria-expanded={open}
            >
              {open ? <X size={18} /> : <Menu size={18} />}
            </button>
          </div>
        </div>
      </Container>

      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0, y: -12 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -12 }}
            transition={{ duration: 0.25, ease: [0.16, 1, 0.3, 1] }}
            className="lg:hidden"
          >
            <Container>
              <div className="glass mt-2 flex flex-col gap-1 rounded-3xl p-4 shadow-xl">
                {LINKS.map((link) => (
                  <a
                    key={link.href}
                    href={link.href}
                    onClick={() => setOpen(false)}
                    className="rounded-xl px-4 py-3 text-sm text-muted transition-colors hover:bg-surface-2 hover:text-foreground"
                  >
                    {link.label}
                  </a>
                ))}
                <div className="mt-2 flex items-center justify-between px-2">
                  <ThemeToggle />
                  <a
                    href="#ceviri"
                    onClick={() => setOpen(false)}
                    className="rounded-full bg-foreground px-4 py-2 text-sm font-medium text-background"
                  >
                    Hemen dene
                  </a>
                </div>
              </div>
            </Container>
          </motion.div>
        )}
      </AnimatePresence>
    </header>
  );
}
