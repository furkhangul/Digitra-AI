import type { Metadata } from "next";
import Script from "next/script";
import { Inter, Sora } from "next/font/google";
import "./globals.css";
import { ThemeProvider } from "@/components/site/theme-provider";
import { SiteFrame } from "@/components/site/site-frame";

const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
  display: "swap",
});

const sora = Sora({
  variable: "--font-cal",
  subsets: ["latin"],
  display: "swap",
});

export const metadata: Metadata = {
  title: "Digitra — TİD Parmak Alfabesi, 3D Eğitim ve Canlı Pratik",
  description:
    "Türk İşaret Dili parmak alfabesinin 29 harfini 3D el modelleriyle keşfet. Kamerayla pratik yap, tanınan harfleri metne dönüştür ve sesli dinle.",
};

const THEME_INIT = `
(function () {
  try {
    var stored = localStorage.getItem("digitra-theme");
    if (stored === "light" || stored === "dark") {
      document.documentElement.setAttribute("data-theme", stored);
    }
  } catch (e) {}
})();
`;

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html
      lang="tr"
      className={`${inter.variable} ${sora.variable} h-full antialiased`}
      suppressHydrationWarning
    >
      <body className="min-h-full flex flex-col bg-background text-foreground">
        <Script id="digitra-theme-init" strategy="beforeInteractive">
          {THEME_INIT}
        </Script>
        <ThemeProvider>
          <SiteFrame>{children}</SiteFrame>
        </ThemeProvider>
      </body>
    </html>
  );
}
