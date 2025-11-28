'use client';

import { useState, useCallback, useRef, useEffect } from 'react';
import { streamChat, listChatSessions, createChatSession, deleteChatSession } from '../api';
import type { AgenticStreamEvent, SourceRef } from '../types';

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  sources?: SourceRef[];
  timestamp: Date;
  isStreaming?: boolean;
}

export interface SavedChat {
  id: string;
  title: string;
  messages: ChatMessage[];
  createdAt: Date;
  updatedAt: Date;
}

interface ChatState {
  sessionId: string | null;
  messages: ChatMessage[];
  isLoading: boolean;
  error: string | null;
  status: string;
  decisions: { tool: string; reasoning: string }[];
}

const STORAGE_KEY = 'vsm_chat_history';

function generateId(): string {
  return Math.random().toString(36).substring(2, 10);
}

function loadSavedChats(): SavedChat[] {
  if (typeof window === 'undefined') return [];
  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved) {
      const chats = JSON.parse(saved);
      return chats.map((c: SavedChat) => ({
        ...c,
        createdAt: new Date(c.createdAt),
        updatedAt: new Date(c.updatedAt),
        messages: c.messages.map(m => ({
          ...m,
          timestamp: new Date(m.timestamp),
        })),
      }));
    }
  } catch (e) {
    console.error('Failed to load saved chats:', e);
  }
  return [];
}

function saveChatToStorage(chats: SavedChat[]): void {
  if (typeof window === 'undefined') return;
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(chats));
  } catch (e) {
    console.error('Failed to save chats:', e);
  }
}

export function useChat() {
  const [state, setState] = useState<ChatState>({
    sessionId: null,
    messages: [],
    isLoading: false,
    error: null,
    status: '',
    decisions: [],
  });
  
  const [savedChats, setSavedChats] = useState<SavedChat[]>([]);
  const [currentChatId, setCurrentChatId] = useState<string | null>(null);
  
  const abortControllerRef = useRef<AbortController | null>(null);
  
  // Load saved chats on mount
  useEffect(() => {
    setSavedChats(loadSavedChats());
  }, []);
  
  // Save current chat when messages change
  useEffect(() => {
    if (currentChatId && state.messages.length > 0) {
      setSavedChats(prev => {
        const updated = prev.map(chat => {
          if (chat.id === currentChatId) {
            return {
              ...chat,
              messages: state.messages,
              updatedAt: new Date(),
              title: state.messages[0]?.content.slice(0, 50) || 'New Chat',
            };
          }
          return chat;
        });
        saveChatToStorage(updated);
        return updated;
      });
    }
  }, [state.messages, currentChatId]);
  
  const sendMessage = useCallback((message: string) => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    
    // Add user message
    const userMessage: ChatMessage = {
      id: generateId(),
      role: 'user',
      content: message,
      timestamp: new Date(),
    };
    
    // Add placeholder assistant message
    const assistantMessage: ChatMessage = {
      id: generateId(),
      role: 'assistant',
      content: '',
      timestamp: new Date(),
      isStreaming: true,
    };
    
    setState(prev => ({
      ...prev,
      messages: [...prev.messages, userMessage, assistantMessage],
      isLoading: true,
      error: null,
      status: 'Thinking...',
      decisions: [],
    }));
    
    let currentSources: SourceRef[] = [];
    
    const handleEvent = (event: AgenticStreamEvent & { type: 'session'; payload: { is_new: boolean; message_count: number }; session_id: string }) => {
      if (event.type === 'session') {
        setState(prev => ({
          ...prev,
          sessionId: event.session_id,
        }));
      } else if (event.type === 'status') {
        setState(prev => ({
          ...prev,
          status: (event.payload as { message: string }).message,
        }));
      } else if (event.type === 'decision') {
        const payload = event.payload as { tool: string; reasoning: string };
        setState(prev => ({
          ...prev,
          decisions: [...prev.decisions, payload],
          status: `Using ${payload.tool}...`,
        }));
      } else if (event.type === 'response') {
        const payload = event.payload as { text: string; sources?: SourceRef[] };
        currentSources = payload.sources || [];
        
        setState(prev => ({
          ...prev,
          messages: prev.messages.map((m, i) => {
            if (i === prev.messages.length - 1 && m.role === 'assistant') {
              return {
                ...m,
                content: payload.text,
                sources: currentSources,
                isStreaming: false,
              };
            }
            return m;
          }),
        }));
      } else if (event.type === 'error') {
        const payload = event.payload as { message: string };
        setState(prev => ({
          ...prev,
          error: payload.message,
        }));
      } else if (event.type === 'complete') {
        setState(prev => ({
          ...prev,
          isLoading: false,
          status: '',
          messages: prev.messages.map((m, i) => {
            if (i === prev.messages.length - 1 && m.role === 'assistant') {
              return { ...m, isStreaming: false };
            }
            return m;
          }),
        }));
      }
    };
    
    const handleError = (error: Error) => {
      setState(prev => ({
        ...prev,
        isLoading: false,
        error: error.message,
        status: '',
        messages: prev.messages.map((m, i) => {
          if (i === prev.messages.length - 1 && m.role === 'assistant') {
            return { ...m, content: 'Error: ' + error.message, isStreaming: false };
          }
          return m;
        }),
      }));
    };
    
    const handleComplete = () => {
      setState(prev => ({
        ...prev,
        isLoading: false,
        status: '',
      }));
    };
    
    abortControllerRef.current = streamChat(
      message,
      state.sessionId,
      handleEvent,
      handleError,
      handleComplete,
    );
  }, [state.sessionId]);
  
  const startNewChat = useCallback(() => {
    // Save current chat if it has messages
    if (state.messages.length > 0 && currentChatId) {
      // Already saved via useEffect
    }
    
    // Create new chat
    const newChatId = generateId();
    const newChat: SavedChat = {
      id: newChatId,
      title: 'New Chat',
      messages: [],
      createdAt: new Date(),
      updatedAt: new Date(),
    };
    
    setSavedChats(prev => {
      const updated = [newChat, ...prev];
      saveChatToStorage(updated);
      return updated;
    });
    
    setCurrentChatId(newChatId);
    setState({
      sessionId: null,
      messages: [],
      isLoading: false,
      error: null,
      status: '',
      decisions: [],
    });
  }, [state.messages, currentChatId]);
  
  const loadChat = useCallback((chatId: string) => {
    const chat = savedChats.find(c => c.id === chatId);
    if (chat) {
      setCurrentChatId(chatId);
      setState(prev => ({
        ...prev,
        messages: chat.messages,
        sessionId: null, // Will create new backend session
      }));
    }
  }, [savedChats]);
  
  const deleteChat = useCallback((chatId: string) => {
    setSavedChats(prev => {
      const updated = prev.filter(c => c.id !== chatId);
      saveChatToStorage(updated);
      return updated;
    });
    
    if (currentChatId === chatId) {
      setCurrentChatId(null);
      setState({
        sessionId: null,
        messages: [],
        isLoading: false,
        error: null,
        status: '',
        decisions: [],
      });
    }
  }, [currentChatId]);
  
  const cancel = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      setState(prev => ({
        ...prev,
        isLoading: false,
        status: 'Cancelled',
      }));
    }
  }, []);
  
  return {
    ...state,
    savedChats,
    currentChatId,
    sendMessage,
    startNewChat,
    loadChat,
    deleteChat,
    cancel,
  };
}

