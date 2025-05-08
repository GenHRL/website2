"use client";
import { useEffect } from "react";
import Prism from "prismjs";
import "prismjs/components/prism-python";

export default function PrismHighlighter(): null {
  useEffect(() => {
    Prism.highlightAll();
  }, []);
  return null;
}
