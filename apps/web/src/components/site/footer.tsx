import Link from "next/link";
import { Mail } from "lucide-react";
import { LogoMark } from "./logo";
import { Container } from "@/components/ui/container";

const COLUMNS = [
  {
    title: "Ürün",
    links: [
      { label: "Harf tanıma", href: "#ceviri" },
      { label: "3D eğitim", href: "#egitim" },
      { label: "Neden Digitra", href: "#how-it-works" },
      { label: "Tanıma teknolojisi", href: "#architecture" },
    ],
  },
  {
    title: "Araştırma",
    links: [
      { label: "Doğrulama yaklaşımı", href: "#science" },
      { label: "Veri kaynakları", href: "#datasets" },
      { label: "SSS", href: "#faq" },
    ],
  },
];

export function Footer() {
  return (
    <footer className="border-t border-border">
      <Container className="py-16">
        <div className="grid grid-cols-2 gap-10 sm:grid-cols-3">
          <div>
            <Link href="/" className="flex items-center gap-2 text-base font-semibold tracking-tight">
              <LogoMark className="h-8 w-8" />
              Digitra
            </Link>
            <p className="mt-4 max-w-xs text-sm leading-relaxed text-muted">
              Türk İşaret Dili parmak alfabesini 3D ellerle öğrenmek,
              kamerayla denemek ve harflerden metin oluşturmak için geliştiriliyor.
            </p>
            <div className="mt-5 flex items-center gap-3">
              <a href="mailto:contact@digitra.ai" aria-label="E-posta" className="glass flex h-9 w-9 items-center justify-center rounded-full text-muted transition-colors hover:text-foreground">
                <Mail size={16} />
              </a>
            </div>
          </div>

          {COLUMNS.map((col) => (
            <div key={col.title}>
              <h4 className="text-sm font-semibold">{col.title}</h4>
              <ul className="mt-4 space-y-3">
                {col.links.map((link) => (
                  <li key={link.label}>
                    <a href={link.href} className="text-sm text-muted transition-colors hover:text-foreground">
                      {link.label}
                    </a>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>

        <div className="mt-14 flex flex-col items-center justify-between gap-4 border-t border-border pt-8 text-xs text-muted sm:flex-row">
          <p>© {new Date().getFullYear()} Digitra. Tüm hakları saklıdır.</p>
          <p>TİD parmak alfabesi · 3D eğitim · Geliştirme sürümü</p>
        </div>
      </Container>
    </footer>
  );
}

