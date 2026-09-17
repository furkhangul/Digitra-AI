"use client";

import { ArrowRight } from "lucide-react";
import { Container, Section } from "@/components/ui/container";
import { Reveal } from "@/components/ui/reveal";

export function CTA() {
  return (
    <Section className="pt-0">
      <Container>
        <Reveal className="relative overflow-hidden rounded-[2.5rem] px-8 py-16 text-center sm:px-16" >
          <div
            className="gradient-pan absolute inset-0 -z-10"
            style={{ background: "linear-gradient(120deg, #6d5bff, #9c7bff, #00d4b8, #6d5bff)" }}
          />
          <div className="noise-overlay absolute inset-0 -z-10" />
          <h2 className="text-3xl font-semibold tracking-tight text-white sm:text-5xl">
            Bir harfle başla.
          </h2>
          <p className="mx-auto mt-4 max-w-xl text-base text-white/80 sm:text-lg">
            Ellerini kameraya göster, ilk harfini metne dönüştür.
            Deneyimini ve önerilerini paylaşarak Digitra&apos;nın gelişimine katkıda bulun.
          </p>
          <div className="mt-8 flex flex-wrap items-center justify-center gap-4">
            <a
              href="#ceviri"
              className="group inline-flex items-center gap-2 rounded-full bg-white px-6 py-3.5 text-sm font-medium text-[#0a0a0f] transition-transform duration-300 hover:-translate-y-0.5"
            >
              Harf tanımayı dene
              <ArrowRight size={16} className="transition-transform duration-300 group-hover:translate-x-1" />
            </a>
            <a
              href="mailto:contact@digitra.ai"
              className="inline-flex items-center gap-2 rounded-full border border-white/30 px-6 py-3.5 text-sm font-medium text-white transition-colors duration-300 hover:bg-white/10"
            >
              Geri bildirim paylaş
            </a>
          </div>
        </Reveal>
      </Container>
    </Section>
  );
}
