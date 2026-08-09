"use client";

import { useCallback, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import {
  Conversation,
  ConversationContent,
  ConversationEmptyState,
  ConversationScrollButton,
} from "@/components/ai-elements/conversation";
import {
  Message,
  MessageContent,
  MessageResponse,
} from "@/components/ai-elements/message";
import {
  PromptInput,
  PromptInputBody,
  PromptInputFooter,
  PromptInputSubmit,
  PromptInputTextarea,
  type PromptInputMessage,
} from "@/components/ai-elements/prompt-input";
import {
  Sources,
  SourcesContent,
  SourcesTrigger,
  Source,
} from "@/components/ai-elements/sources";
import { Suggestion } from "@/components/ai-elements/suggestion";
import { Shimmer } from "@/components/ai-elements/shimmer";
import { RetrievalRail } from "@/components/retrieval-rail";
import { streamChat, type ScoredChunk, type TokenUsage } from "@/lib/rag";

type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  text: string;
  sources: string[];
  usage?: TokenUsage;
  failed?: boolean;
};

type Phase = "idle" | "retrieving" | "streaming";

const SUGGESTIONS = [
  "When do I need original receipts for expense claims?",
  "How many days of bereavement leave do I get?",
  "What are the VPN device requirements for macOS?",
  "How does the 2026 pricing differ from 2025?",
];

const spring = { type: "spring", stiffness: 100, damping: 20 } as const;

export function ChatShell() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [chunks, setChunks] = useState<ScoredChunk[]>([]);
  const [phase, setPhase] = useState<Phase>("idle");
  const [error, setError] = useState<string | null>(null);
  const counter = useRef(0);

  const isBusy = phase !== "idle";

  const send = useCallback(
    async (question: string) => {
      if (!question.trim() || isBusy) return;
      setError(null);
      const userMsg: ChatMessage = {
        id: `m${++counter.current}`,
        role: "user",
        text: question,
        sources: [],
      };
      const botId = `m${++counter.current}`;
      setMessages((prev) => [
        ...prev,
        userMsg,
        { id: botId, role: "assistant", text: "", sources: [] },
      ]);
      setChunks([]);
      setPhase("retrieving");

      const patch = (fn: (m: ChatMessage) => Partial<ChatMessage>) =>
        setMessages((prev) =>
          prev.map((m) => (m.id === botId ? { ...m, ...fn(m) } : m)),
        );

      try {
        const history = messages
          .filter((m) => m.text && !m.failed)
          .slice(-6)
          .map((m) => ({ role: m.role, content: m.text }));
        await streamChat(question, (evt) => {
          if (evt.type === "retrieval") {
            setChunks(evt.chunks);
            setPhase("streaming");
          } else if (evt.type === "text-delta") {
            setMessages((prev) =>
              prev.map((m) =>
                m.id === botId ? { ...m, text: m.text + evt.text } : m,
              ),
            );
          } else if (evt.type === "sources") {
            patch(() => ({ sources: evt.items }));
          } else if (evt.type === "usage") {
            patch(() => ({
              usage: {
                prompt_tokens: evt.prompt_tokens,
                completion_tokens: evt.completion_tokens,
              },
            }));
          }
        }, history);
      } catch (e) {
        patch(() => ({
          failed: true,
          text: "The assistant could not reach the backend. Check that it is running and try again.",
        }));
        setError(e instanceof Error ? e.message : "Unknown error");
      } finally {
        setPhase("idle");
      }
    },
    [isBusy],
  );

  return (
    <div className="grid min-h-0 flex-1 grid-cols-1 lg:grid-cols-[minmax(0,1fr)_320px]">
      <div className="flex min-h-0 flex-col">
        {error && (
          <div className="border-b border-destructive/30 bg-destructive/5 px-4 py-2 text-sm text-destructive">
            {error}
          </div>
        )}
        <Conversation className="min-h-0 flex-1">
          <ConversationContent className="mx-auto w-full max-w-3xl gap-6 px-4 py-6">
            {messages.length === 0 ? (
              <ConversationEmptyState className="gap-4">
                <p className="text-sm text-muted-foreground">
                  What can I look up for you?
                </p>
                <div className="flex max-w-lg flex-wrap items-center justify-center gap-2">
                  {SUGGESTIONS.map((s) => (
                    <Suggestion key={s} suggestion={s} onClick={send} />
                  ))}
                </div>
              </ConversationEmptyState>
            ) : (
              <AnimatePresence initial={false}>
                {messages.map((m) => (
                  <motion.div
                    key={m.id}
                    initial={{ opacity: 0, y: 12 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={spring}
                  >
                    <Message from={m.role}>
                      <MessageContent>
                        {m.role === "assistant" ? (
                          m.text ? (
                            <MessageResponse>{m.text}</MessageResponse>
                          ) : (
                            <Shimmer className="text-sm">
                              {phase === "retrieving"
                                ? "Searching the knowledge base…"
                                : "Writing…"}
                            </Shimmer>
                          )
                        ) : (
                          m.text
                        )}
                        {m.role === "assistant" && m.sources.length > 0 && (
                          <Sources>
                            <SourcesTrigger count={m.sources.length} />
                            <SourcesContent>
                              {m.sources.map((s) => (
                                <Source key={s} title={s} />
                              ))}
                          </SourcesContent>
                          </Sources>
                        )}
                        {m.role === "assistant" && m.usage && (
                          <p className="mt-1 font-mono text-[11px] text-muted-foreground">
                            {m.usage.prompt_tokens}p + {m.usage.completion_tokens}c tokens
                          </p>
                        )}
                      </MessageContent>
                    </Message>
                  </motion.div>
                ))}
              </AnimatePresence>
            )}
          </ConversationContent>
          <ConversationScrollButton />
        </Conversation>
        <div className="border-t border-border bg-background px-4 py-3">
          <div className="mx-auto w-full max-w-3xl">
            <PromptInput
              onSubmit={(msg: PromptInputMessage) => void send(msg.text)}
            >
              <PromptInputBody>
                <PromptInputTextarea
                  placeholder={
                    isBusy ? "Working…" : "Ask about policies, pricing, IT guides…"
                  }
                  disabled={isBusy}
                />
              </PromptInputBody>
              <PromptInputFooter className="justify-end">
                <PromptInputSubmit
                  status={phase === "streaming" ? "streaming" : isBusy ? "submitted" : "ready"}
                />
              </PromptInputFooter>
            </PromptInput>
          </div>
        </div>
      </div>
      <aside className="hidden min-h-0 border-l border-border lg:block">
        <RetrievalRail chunks={chunks} phase={phase} />
      </aside>
    </div>
  );
}
