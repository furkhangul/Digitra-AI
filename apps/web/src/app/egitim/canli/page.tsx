import type { Metadata } from "next";
import { FingerspellingTraining } from "@/components/home/fingerspelling-training";

export const metadata: Metadata = {
  title: "Digitra — TİD Parmak Alfabesiyle Canlı Pratik",
  description: "3D el modelini izle, harfi kameraya göster. TİD parmak alfabesiyle kısa kelimelerden hareketli harflere kadar kendi hızında pratik yap.",
};

export default function LiveTrainingPage() {
  return <FingerspellingTraining />;
}
