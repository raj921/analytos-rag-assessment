"use client";

import { AnimatePresence, motion } from "framer-motion";
import type { ScoredChunk } from "@/lib/rag";

type Phase = "idle" | "retrieving" | "streaming";

const spring = { type: "spring", stiffness: 100, damping: 20 } as const;

export function RetrievalRail({ chunks, phase }: { chunks: ScoredChunk[]; phase: Phase }) {
  const max = Math.max(4, ...chunks.map((c) => c.score));
  return (
    <div className="flex h-full min-h-0 flex-col overflow-y-auto p-4">
      <div className="mb-3 flex items-baseline justify-between">
        <h2 className="text-xs font-medium uppercase tracking-widest text-muted-foreground">
          Retrieval inspector
        </h2>
        <span className="font-mono text-[10px] text-muted-foreground">
          hybrid + semantic
        </span>
      </div>
      <ol className="flex flex-col divide-y divide-border">
        <AnimatePresence initial={false}>
          {chunks.map((c, i) => (
            <motion.li
              key={`${c.title}-${i}`}
              initial={{ opacity: 0, x: 16 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ ...spring, delay: i * 0.06 }}
              className="py-2.5"
            >
              <div className="flex items-center justify-between gap-2">
                <span className="truncate text-sm">{c.title}</span>
                <span className="font-mono text-xs tabular-nums text-muted-foreground">
                  {c.score.toFixed(2)}
                </span>
              </div>
              <div className="mt-1.5 h-1 w-full overflow-hidden rounded-full bg-muted">
                <motion.div
                  className="h-full rounded-full bg-emerald-600/80"
                  initial={{ width: 0 }}
                  animate={{ width: `${Math.min(100, (c.score / max) * 100)}%` }}
                  transition={spring}
                />
              </div>
            </motion.li>
          ))}
        </AnimatePresence>
      </ol>
    </div>
  );
}
