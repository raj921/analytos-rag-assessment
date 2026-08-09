export type ScoredChunk = { title: string; score: number };
export type TokenUsage = { prompt_tokens: number; completion_tokens: number };

export type StreamEvent =
  | {
      type: "retrieval";
      chunks: ScoredChunk[];
      query?: string;
      confidence?: string;
      cached?: boolean;
    }
  | { type: "text-delta"; text: string }
  | { type: "sources"; items: string[] }
  | { type: "usage"; prompt_tokens: number; completion_tokens: number }
  | { type: "done" }
  | { type: "error"; message: string };

export const API_BASE = process.env.NEXT_PUBLIC_RAG_API ?? "";

export type HistoryTurn = { role: "user" | "assistant"; content: string };

export async function streamChat(
  question: string,
  onEvent: (event: StreamEvent) => void,
  history: HistoryTurn[] = [],
): Promise<void> {
  const res = await fetch(`${API_BASE}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, history }),
  });
  if (!res.ok || !res.body) throw new Error(`Backend error: ${res.status}`);

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const frames = buffer.split("\n\n");
    buffer = frames.pop() ?? "";
    for (const frame of frames) {
      const line = frame.trim();
      if (!line.startsWith("data:")) continue;
      onEvent(JSON.parse(line.slice(5)) as StreamEvent);
    }
  }
}
