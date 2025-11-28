'use client';

import { useState, useRef, useEffect } from 'react';
import Link from 'next/link';
import { useChat, ChatMessage, SavedChat } from '@/lib/hooks/useChat';
import { SourceRef } from '@/lib/types';

// Chat message bubble component
function MessageBubble({ message }: { message: ChatMessage }) {
  const isUser = message.role === 'user';
  
  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-4`}>
      <div
        className={`max-w-[80%] rounded-2xl px-4 py-3 ${
          isUser
            ? 'bg-indigo-600 text-white'
            : 'bg-zinc-800 text-zinc-100 border border-zinc-700'
        }`}
      >
        {message.isStreaming && !message.content ? (
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 bg-indigo-400 rounded-full animate-pulse" />
            <div className="w-2 h-2 bg-indigo-400 rounded-full animate-pulse delay-75" />
            <div className="w-2 h-2 bg-indigo-400 rounded-full animate-pulse delay-150" />
          </div>
        ) : (
          <>
            <p className="whitespace-pre-wrap text-sm leading-relaxed">{message.content}</p>
            {message.sources && message.sources.length > 0 && (
              <div className="mt-3 pt-3 border-t border-zinc-600">
                <p className="text-xs text-zinc-400 mb-2">Sources:</p>
                <div className="flex flex-wrap gap-2">
                  {message.sources.map((source, idx) => (
                    <SourceBadge key={idx} source={source} />
                  ))}
                </div>
              </div>
            )}
          </>
        )}
        <p className="text-xs text-zinc-400 mt-2">
          {message.timestamp.toLocaleTimeString()}
        </p>
      </div>
    </div>
  );
}

function SourceBadge({ source }: { source: SourceRef }) {
  return (
    <span className="inline-flex items-center gap-1 px-2 py-1 bg-zinc-700 rounded text-xs">
      <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
      </svg>
      <span>Page {source.page}</span>
      {source.score && (
        <span className="text-zinc-500">({(source.score * 100).toFixed(0)}%)</span>
      )}
    </span>
  );
}

// Sidebar with chat history
function ChatSidebar({
  savedChats,
  currentChatId,
  onNewChat,
  onLoadChat,
  onDeleteChat,
}: {
  savedChats: SavedChat[];
  currentChatId: string | null;
  onNewChat: () => void;
  onLoadChat: (id: string) => void;
  onDeleteChat: (id: string) => void;
}) {
  return (
    <div className="w-64 bg-zinc-900 border-r border-zinc-800 flex flex-col h-full">
      {/* Header */}
      <div className="p-4 border-b border-zinc-800">
        <button
          onClick={onNewChat}
          className="w-full flex items-center justify-center gap-2 px-4 py-2.5 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg transition-colors font-medium"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
          </svg>
          New Chat
        </button>
      </div>
      
      {/* Chat list */}
      <div className="flex-1 overflow-y-auto p-2">
        {savedChats.length === 0 ? (
          <p className="text-zinc-500 text-sm text-center py-4">No saved chats</p>
        ) : (
          <div className="space-y-1">
            {savedChats.map(chat => (
              <div
                key={chat.id}
                className={`group flex items-center gap-2 px-3 py-2 rounded-lg cursor-pointer transition-colors ${
                  currentChatId === chat.id
                    ? 'bg-zinc-800 text-white'
                    : 'text-zinc-400 hover:bg-zinc-800/50 hover:text-white'
                }`}
                onClick={() => onLoadChat(chat.id)}
              >
                <svg className="w-4 h-4 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
                </svg>
                <span className="flex-1 truncate text-sm">{chat.title}</span>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    onDeleteChat(chat.id);
                  }}
                  className="opacity-0 group-hover:opacity-100 p-1 hover:bg-zinc-700 rounded transition-opacity"
                >
                  <svg className="w-4 h-4 text-zinc-400 hover:text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                  </svg>
                </button>
              </div>
            ))}
          </div>
        )}
      </div>
      
      {/* Back to search link */}
      <div className="p-4 border-t border-zinc-800">
        <Link
          href="/"
          className="flex items-center gap-2 text-zinc-400 hover:text-white text-sm transition-colors"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
          </svg>
          Back to Search
        </Link>
      </div>
    </div>
  );
}

// Decision trail component
function DecisionTrail({ decisions }: { decisions: { tool: string; reasoning: string }[] }) {
  if (decisions.length === 0) return null;
  
  return (
    <div className="bg-zinc-800/50 border border-zinc-700 rounded-lg p-3 mb-4">
      <p className="text-xs text-zinc-400 mb-2 font-medium">Agent thinking:</p>
      <div className="space-y-2">
        {decisions.map((d, i) => (
          <div key={i} className="flex items-start gap-2 text-xs">
            <span className="px-1.5 py-0.5 bg-indigo-600/30 text-indigo-300 rounded font-mono">
              {d.tool}
            </span>
            <span className="text-zinc-400">{d.reasoning}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function ChatPage() {
  const {
    messages,
    isLoading,
    error,
    status,
    decisions,
    savedChats,
    currentChatId,
    sendMessage,
    startNewChat,
    loadChat,
    deleteChat,
    cancel,
  } = useChat();
  
  const [input, setInput] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  
  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, decisions]);
  
  // Focus input on load
  useEffect(() => {
    inputRef.current?.focus();
  }, []);
  
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (input.trim() && !isLoading) {
      sendMessage(input.trim());
      setInput('');
    }
  };
  
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };
  
  return (
    <div className="flex h-screen bg-zinc-950 text-white">
      {/* Sidebar */}
      <ChatSidebar
        savedChats={savedChats}
        currentChatId={currentChatId}
        onNewChat={startNewChat}
        onLoadChat={loadChat}
        onDeleteChat={deleteChat}
      />
      
      {/* Main chat area */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <header className="px-6 py-4 border-b border-zinc-800 flex items-center justify-between">
          <div>
            <h1 className="text-lg font-semibold">VSM Chat</h1>
            <p className="text-sm text-zinc-400">Ask questions about your manuals</p>
          </div>
          {status && (
            <div className="flex items-center gap-2 text-sm text-zinc-400">
              <div className="w-2 h-2 bg-indigo-500 rounded-full animate-pulse" />
              {status}
            </div>
          )}
        </header>
        
        {/* Messages area */}
        <div className="flex-1 overflow-y-auto p-6">
          {messages.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center text-center">
              <div className="w-16 h-16 bg-indigo-600/20 rounded-2xl flex items-center justify-center mb-4">
                <svg className="w-8 h-8 text-indigo-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
                </svg>
              </div>
              <h2 className="text-xl font-semibold mb-2">Start a conversation</h2>
              <p className="text-zinc-400 max-w-md">
                Ask questions about your technical manuals. The agent will search through documents and provide answers with sources.
              </p>
              <div className="mt-6 flex flex-wrap gap-2 justify-center">
                {[
                  'What are the jumper settings?',
                  'Show me the wiring diagram',
                  'How do I reset the alarm?',
                ].map((suggestion) => (
                  <button
                    key={suggestion}
                    onClick={() => sendMessage(suggestion)}
                    className="px-4 py-2 bg-zinc-800 hover:bg-zinc-700 rounded-lg text-sm text-zinc-300 transition-colors"
                  >
                    {suggestion}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="max-w-3xl mx-auto">
              {messages.map((message) => (
                <MessageBubble key={message.id} message={message} />
              ))}
              
              {/* Show decisions while loading */}
              {isLoading && decisions.length > 0 && (
                <DecisionTrail decisions={decisions} />
              )}
              
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>
        
        {/* Error display */}
        {error && (
          <div className="mx-6 mb-4 p-3 bg-red-900/30 border border-red-800 rounded-lg text-red-300 text-sm">
            {error}
          </div>
        )}
        
        {/* Input area */}
        <div className="p-4 border-t border-zinc-800">
          <form onSubmit={handleSubmit} className="max-w-3xl mx-auto">
            <div className="relative flex items-end gap-2">
              <textarea
                ref={inputRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Ask a question..."
                rows={1}
                className="flex-1 px-4 py-3 bg-zinc-800 border border-zinc-700 rounded-xl text-white placeholder-zinc-500 resize-none focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
                style={{ minHeight: '48px', maxHeight: '200px' }}
              />
              {isLoading ? (
                <button
                  type="button"
                  onClick={cancel}
                  className="px-4 py-3 bg-red-600 hover:bg-red-500 rounded-xl transition-colors"
                >
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              ) : (
                <button
                  type="submit"
                  disabled={!input.trim()}
                  className="px-4 py-3 bg-indigo-600 hover:bg-indigo-500 disabled:bg-zinc-700 disabled:cursor-not-allowed rounded-xl transition-colors"
                >
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
                  </svg>
                </button>
              )}
            </div>
            <p className="text-xs text-zinc-500 mt-2 text-center">
              Press Enter to send, Shift+Enter for new line
            </p>
          </form>
        </div>
      </div>
    </div>
  );
}

