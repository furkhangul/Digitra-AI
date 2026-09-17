import { Intro } from "@/components/home/intro";
import { Hero } from "@/components/home/hero";
import { StackingCards } from "@/components/home/stacking-cards";
import { HowItWorks } from "@/components/home/how-it-works";
import { Ceviri } from "@/components/home/ceviri";
import { Egitim } from "@/components/home/egitim";
import { Architecture } from "@/components/home/architecture";
import { LiveStatistics } from "@/components/home/live-statistics";
import { ScientificValidation } from "@/components/home/scientific-validation";
import { Accessibility } from "@/components/home/accessibility";
import { Datasets } from "@/components/home/datasets";
import { FAQ } from "@/components/home/faq";
import { CTA } from "@/components/home/cta";

export default function Home() {
  return (
    <>
      <Intro />
      <Hero />
      <StackingCards />
      <HowItWorks />
      <Ceviri />
      <Egitim />
      <Architecture />
      <LiveStatistics />
      <ScientificValidation />
      <Accessibility />
      <Datasets />
      <FAQ />
      <CTA />
    </>
  );
}
