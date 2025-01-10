"use client";

import { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import {
  ChatBubble,
  ChatBubbleAvatar,
  ChatBubbleMessage,
} from "@/app/ui/chat/chat-bubble";
import { ChatInput } from "@/app/ui/chat/chat-input";
import { ChatMessageList } from "@/app/ui/chat/chat-message-list";
import { Button } from "@/components/ui/button";
import { CornerDownLeft } from "lucide-react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import CodeDisplayBlock from "@/app/ui/code-display-block";

interface Message {
  sender: "user" | "assistant";
  text: string;
}

interface ChatbotResponse {
  response: string;
}

export default function Home() {
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState<Message[]>([]);
  const [isGenerating, setIsGenerating] = useState(false);
  const messagesRef = useRef<HTMLDivElement>(null);
  const formRef = useRef<HTMLFormElement>(null);

  useEffect(() => {
    if (messagesRef.current) {
      messagesRef.current.scrollTop = messagesRef.current.scrollHeight;
    }
  }, [messages]);

  const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value);
  };

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (!input.trim()) return;

    const userMessage: Message = { sender: "user", text: input };
    setMessages([...messages, userMessage]);
    setInput('');
    setIsGenerating(true);

    try {
      const response = await axios.post<ChatbotResponse>('http://localhost:5000/api/chatbot', { input });
      const botMessage: Message = { sender: "assistant", text: response.data.response };
      setMessages([...messages, userMessage, botMessage]);
    } catch (error) {
      console.error('Error fetching response:', error);
    } finally {
      setIsGenerating(false);
    }
  };

  const onKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (isGenerating || !input) return;
      setIsGenerating(true);
      handleSubmit(e as unknown as React.FormEvent<HTMLFormElement>);
    }
  };

  return (
    <main className="flex h-screen w-full max-w-3xl flex-col items-center mx-auto py-6 no-scrollbar scrollbar-none">
      <ChatMessageList ref={messagesRef}>
        {messages.map((message, index) => (
          <ChatBubble
            key={index}
            variant={message.sender === "user" ? "sent" : "received"}
          >
            <ChatBubbleAvatar
              src=""
              fallback={message.sender === "user" ? "👨🏽" : "🤖"}
            />
            <ChatBubbleMessage>
              {message.text.split("```").map((part, index) => (
                index % 2 === 0 ? (
                  <Markdown key={index} remarkPlugins={[remarkGfm]}>
                    {part}
                  </Markdown>
                ) : (
                  <pre className="whitespace-pre-wrap pt-2" key={index}>
                    <CodeDisplayBlock code={part} lang="" />
                  </pre>
                )
              ))}
            </ChatBubbleMessage>
          </ChatBubble>
        ))}
        {isGenerating && (
          <ChatBubble variant="received">
            <ChatBubbleAvatar src="" fallback="🤖" />
            <ChatBubbleMessage isLoading />
          </ChatBubble>
        )}
      </ChatMessageList>
      <div className="w-full px-4">
        <form
          ref={formRef}
          onSubmit={handleSubmit}
          className="relative rounded-lg border bg-background focus-within:ring-1 focus-within:ring-ring flex items-center"
        >
          <ChatInput
            value={input}
            onKeyDown={onKeyDown}
            onChange={handleInputChange}
            placeholder="Type your message here..."
            className="min-h-12 resize-none rounded-lg bg-background border-0 p-3 shadow-none focus-visible:ring-0 flex-grow"
          />
          <Button
            disabled={!input || isGenerating}
            type="submit"
            size="sm"
            className="ml-2 gap-1.5 flex align-center mr-2"
          >
            Send Message
            <CornerDownLeft className="size-3.5" />
          </Button>
        </form>
      </div>
    </main>
  );
}