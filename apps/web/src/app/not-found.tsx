import Link from "next/link";
import { ArrowLeft, Hand } from "lucide-react";
import { Container } from "@/components/ui/container";

export default function NotFound() {
  return (
    <section className="mesh-bg flex min-h-[80vh] items-center py-32">
      <Container className="text-center">
        <span className="mx-auto flex h-14 w-14 items-center justify-center rounded-2xl bg-primary/10 text-primary">
          <Hand size={24} />
        </span>
        <p className="mt-6 text-xs font-medium uppercase tracking-[0.2em] text-primary">404</p>
        <h1 className="mt-3 text-4xl font-semibold tracking-tight sm:text-5xl">Bu sayfa bulunamadı.</h1>
        <p className="mx-auto mt-4 max-w-md leading-relaxed text-muted">
          Aradığın adres taşınmış ya da hiç var olmamış olabilir. Harf tanımaya ya da 3D eğitime
          buradan devam edebilirsin.
        </p>
        <div className="mt-8 flex flex-wrap items-center justify-center gap-3">
          <Link href="/" className="inline-flex items-center gap-2 rounded-full bg-foreground px-5 py-3 text-sm font-medium text-background">
            <ArrowLeft size={16} /> Ana sayfaya dön
          </Link>
          <Link href="/#ceviri" className="inline-flex items-center gap-2 rounded-full border border-border px-5 py-3 text-sm font-medium hover:border-primary/40">
            Harf tanımayı dene
          </Link>
        </div>
      </Container>
    </section>
  );
}
