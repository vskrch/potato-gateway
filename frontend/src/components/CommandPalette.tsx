import React, { useState, useEffect, useMemo, useRef } from 'react'
import {
  Search,
  LayoutDashboard,
  BarChart3,
  ListFilter,
  Radio,
  BrainCircuit,
  Coins,
  Terminal,
  MessageSquare,
  Key,
  Users,
  Server,
  Activity,
  Cpu,
  GitFork,
  Layers,
  Filter,
  Zap,
  RefreshCw,
  Copy,
  Check,
  X
} from 'lucide-react'

interface CommandPaletteProps {
  open: boolean
  onClose: () => void
  onNavigate: (pageId: string) => void
  onRefreshCatalog?: () => void
  isAdmin?: boolean
}

interface CommandItem {
  id: string
  title: string
  subtitle: string
  category: 'Navigation' | 'Actions' | 'Tools'
  icon: React.ComponentType<{ className?: string }>
  badge?: string
  action: () => void
}

export default function CommandPalette({
  open,
  onClose,
  onNavigate,
  onRefreshCatalog,
  isAdmin,
}: CommandPaletteProps) {
  const [query, setQuery] = useState('')
  const [selectedIndex, setSelectedIndex] = useState(0)
  const [copiedUrl, setCopiedUrl] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    if (open) {
      setQuery('')
      setSelectedIndex(0)
      setTimeout(() => inputRef.current?.focus(), 50)
    }
  }, [open])

  const copyBaseUrl = () => {
    const url = `${window.location.origin}/v1`
    navigator.clipboard.writeText(url)
    setCopiedUrl(true)
    setTimeout(() => {
      setCopiedUrl(false)
      onClose()
    }, 800)
  }

  const items: CommandItem[] = useMemo(() => {
    const list: CommandItem[] = [
      // Quick Actions
      {
        id: 'action-copy-url',
        title: 'Copy OpenAI Base URL',
        subtitle: `${window.location.origin}/v1 (Use in Cursor, Cline, SDKs)`,
        category: 'Actions',
        icon: copiedUrl ? Check : Copy,
        badge: 'Base URL',
        action: copyBaseUrl,
      },
      ...(isAdmin && onRefreshCatalog
        ? [
            {
              id: 'action-refresh',
              title: 'Trigger Catalog & Model Ladder Re-rank',
              subtitle: 'Poll live health and update multi-tier fallback scores',
              category: 'Actions' as const,
              icon: RefreshCw,
              badge: 'Admin',
              action: () => {
                onRefreshCatalog()
                onClose()
              },
            },
          ]
        : []),

      // Tools & Developer
      {
        id: 'page-playground',
        title: 'API Playground',
        subtitle: 'Interactive test prompt console with streaming TTFT',
        category: 'Tools',
        icon: Terminal,
        badge: 'Interactive',
        action: () => {
          onNavigate('playground')
          onClose()
        },
      },
      {
        id: 'page-chat',
        title: 'Full-screen Chat',
        subtitle: 'Standalone Claude-style multi-turn chat experience',
        category: 'Tools',
        icon: MessageSquare,
        badge: 'Full UI',
        action: () => {
          onNavigate('chat')
          onClose()
        },
      },
      {
        id: 'page-account',
        title: 'API Keys & Account',
        subtitle: 'Manage client tokens, usage quotas, and access credentials',
        category: 'Tools',
        icon: Key,
        action: () => {
          onNavigate('account')
          onClose()
        },
      },

      // Navigation - Core & Analytics
      {
        id: 'page-dashboard',
        title: 'System Overview',
        subtitle: 'Gateway uptime, active providers, and fallback switches',
        category: 'Navigation',
        icon: LayoutDashboard,
        badge: '99.99% HA',
        action: () => {
          onNavigate('dashboard')
          onClose()
        },
      },
      {
        id: 'page-analytics',
        title: 'Analytics Center',
        subtitle: 'Throughput charts, P95/P99 latency, and token metrics',
        category: 'Navigation',
        icon: BarChart3,
        action: () => {
          onNavigate('analytics')
          onClose()
        },
      },
      {
        id: 'page-requests',
        title: 'Request Explorer',
        subtitle: 'Inspect live traces, fallback attempts, and decision trees',
        category: 'Navigation',
        icon: ListFilter,
        action: () => {
          onNavigate('requests')
          onClose()
        },
      },
      {
        id: 'page-live',
        title: 'Live Event Stream',
        subtitle: 'Real-time SSE event pipeline feed',
        category: 'Navigation',
        icon: Radio,
        badge: 'Real-time',
        action: () => {
          onNavigate('live')
          onClose()
        },
      },
      {
        id: 'page-intents',
        title: 'Intent Intelligence',
        subtitle: '6-intent classification breakdown & model affinity',
        category: 'Navigation',
        icon: BrainCircuit,
        action: () => {
          onNavigate('intents')
          onClose()
        },
      },
      {
        id: 'page-cost',
        title: 'Cost & Savings',
        subtitle: 'Token spend vs direct API costs & savings calculator',
        category: 'Navigation',
        icon: Coins,
        action: () => {
          onNavigate('cost')
          onClose()
        },
      },

      // Navigation - Admin
      ...(isAdmin
        ? [
            {
              id: 'page-health',
              title: 'Provider Health & Circuit Breakers',
              subtitle: 'Two-tier breaker states, cooldowns, and error rates',
              category: 'Navigation' as const,
              icon: Activity,
              badge: 'Two-Tier',
              action: () => {
                onNavigate('health')
                onClose()
              },
            },
            {
              id: 'page-providers',
              title: 'LLM Providers',
              subtitle: 'Manage NIM, Groq, Together, Cerebras & Ollama runtimes',
              category: 'Navigation' as const,
              icon: Server,
              action: () => {
                onNavigate('providers')
                onClose()
              },
            },
            {
              id: 'page-models',
              title: 'Model Catalog',
              subtitle: 'Live model inventory, ELO scores, and parameter sizes',
              category: 'Navigation' as const,
              icon: Cpu,
              action: () => {
                onNavigate('models')
                onClose()
              },
            },
            {
              id: 'page-routing',
              title: 'Routing Strategy',
              subtitle: 'Bandit router preferences, intent chains & rules',
              category: 'Navigation' as const,
              icon: GitFork,
              action: () => {
                onNavigate('routing')
                onClose()
              },
            },
            {
              id: 'page-ladders',
              title: 'Model Ladders',
              subtitle: 'Drag-and-drop fallback chains per virtual model',
              category: 'Navigation' as const,
              icon: Layers,
              action: () => {
                onNavigate('ladders')
                onClose()
              },
            },
            {
              id: 'page-pools',
              title: 'Model Pool Gating',
              subtitle: 'Per-model intent gating & auto-router inclusion',
              category: 'Navigation' as const,
              icon: Filter,
              action: () => {
                onNavigate('pools')
                onClose()
              },
            },
            {
              id: 'page-rl',
              title: 'Adaptive RL Engine',
              subtitle: 'LinUCB contextual bandit feature weights and rewards',
              category: 'Navigation' as const,
              icon: Zap,
              badge: 'LinUCB',
              action: () => {
                onNavigate('rl')
                onClose()
              },
            },
            {
              id: 'page-users',
              title: 'User Management',
              subtitle: 'Manage user access levels, approvals, and API keys',
              category: 'Navigation' as const,
              icon: Users,
              action: () => {
                onNavigate('users')
                onClose()
              },
            },
          ]
        : []),
    ]

    if (!query.trim()) return list
    const q = query.toLowerCase()
    return list.filter(
      item =>
        item.title.toLowerCase().includes(q) ||
        item.subtitle.toLowerCase().includes(q) ||
        item.category.toLowerCase().includes(q) ||
        (item.badge && item.badge.toLowerCase().includes(q))
    )
  }, [query, copiedUrl, isAdmin, onRefreshCatalog, onNavigate, onClose])

  useEffect(() => {
    setSelectedIndex(0)
  }, [items.length])

  // Keyboard navigation
  useEffect(() => {
    if (!open) return
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'ArrowDown') {
        e.preventDefault()
        setSelectedIndex(prev => (prev + 1) % Math.max(1, items.length))
      } else if (e.key === 'ArrowUp') {
        e.preventDefault()
        setSelectedIndex(prev => (prev - 1 + items.length) % Math.max(1, items.length))
      } else if (e.key === 'Enter') {
        e.preventDefault()
        if (items[selectedIndex]) {
          items[selectedIndex].action()
        }
      } else if (e.key === 'Escape') {
        e.preventDefault()
        onClose()
      }
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [open, items, selectedIndex, onClose])

  if (!open) return null

  return (
    <div
      className="fixed inset-0 z-50 flex items-start justify-center pt-16 sm:pt-24 px-4 bg-black/75 backdrop-blur-md animate-[fadeIn_0.15s_ease-out]"
      onClick={onClose}
    >
      <div
        className="w-full max-w-xl bg-zinc-950/95 border border-white/[0.12] rounded-2xl shadow-[0_20px_60px_-15px_rgba(0,0,0,0.8),0_0_30px_rgba(139,92,246,0.15)] overflow-hidden flex flex-col max-h-[75vh]"
        onClick={e => e.stopPropagation()}
      >
        {/* Search Input Bar */}
        <div className="flex items-center gap-3 px-4 py-3.5 border-b border-white/[0.08] bg-white/[0.02]">
          <Search className="w-5 h-5 text-zinc-400 shrink-0" />
          <input
            ref={inputRef}
            type="text"
            value={query}
            onChange={e => setQuery(e.target.value)}
            placeholder="Type a command or jump to page..."
            className="flex-1 bg-transparent text-sm text-white placeholder-zinc-500 focus:outline-none font-medium"
          />
          {query && (
            <button
              onClick={() => setQuery('')}
              className="p-1 text-zinc-400 hover:text-white rounded-md transition-colors"
            >
              <X className="w-4 h-4" />
            </button>
          )}
          <kbd className="hidden sm:inline-block px-2 py-0.5 text-[10px] font-mono font-semibold text-zinc-400 bg-white/[0.06] border border-white/[0.1] rounded">
            ESC
          </kbd>
        </div>

        {/* Results List */}
        <div className="flex-1 overflow-y-auto p-2 custom-scrollbar space-y-1">
          {items.length === 0 ? (
            <div className="p-8 text-center text-zinc-500 text-xs flex flex-col items-center gap-2">
              <Search className="w-6 h-6 text-zinc-600 stroke-1" />
              <span>No matching commands or pages found</span>
            </div>
          ) : (
            items.map((item, idx) => {
              const Icon = item.icon
              const isSelected = idx === selectedIndex
              return (
                <button
                  key={item.id}
                  onClick={item.action}
                  onMouseEnter={() => setSelectedIndex(idx)}
                  className={`w-full flex items-center justify-between gap-3 px-3.5 py-2.5 rounded-xl text-left transition-all duration-100 ${
                    isSelected
                      ? 'bg-violet-600/20 text-white border border-violet-500/40 shadow-[0_0_15px_rgba(139,92,246,0.15)]'
                      : 'text-zinc-300 hover:bg-white/[0.04] border border-transparent'
                  }`}
                >
                  <div className="flex items-center gap-3 min-w-0">
                    <div
                      className={`p-2 rounded-lg shrink-0 ${
                        isSelected
                          ? 'bg-violet-500/20 text-violet-300 border border-violet-500/30'
                          : 'bg-white/[0.04] text-zinc-400 border border-white/[0.06]'
                      }`}
                    >
                      <Icon className="w-4 h-4" />
                    </div>
                    <div className="min-w-0">
                      <div className="flex items-center gap-2">
                        <span className="text-xs font-semibold truncate text-white">
                          {item.title}
                        </span>
                        {item.badge && (
                          <span className="text-[10px] font-medium font-mono px-1.5 py-0.2 rounded bg-violet-500/15 text-violet-300 border border-violet-500/30">
                            {item.badge}
                          </span>
                        )}
                      </div>
                      <p className="text-[11px] text-zinc-400 truncate mt-0.5">{item.subtitle}</p>
                    </div>
                  </div>
                  <span className="text-[10px] font-mono text-zinc-400 uppercase tracking-wider shrink-0 hidden sm:inline">
                    {item.category}
                  </span>
                </button>
              )
            })
          )}
        </div>

        {/* Footer shortcuts */}
        <div className="px-4 py-2.5 border-t border-white/[0.08] bg-zinc-950/80 flex items-center justify-between text-[11px] text-zinc-400">
          <div className="flex items-center gap-3">
            <span className="flex items-center gap-1">
              <kbd className="px-1.5 py-0.5 rounded bg-white/[0.06] border border-white/[0.1] text-[10px] font-mono text-zinc-300">
                ↑
              </kbd>
              <kbd className="px-1.5 py-0.5 rounded bg-white/[0.06] border border-white/[0.1] text-[10px] font-mono text-zinc-300">
                ↓
              </kbd>
              <span className="text-zinc-400 ml-1">Navigate</span>
            </span>
            <span className="flex items-center gap-1">
              <kbd className="px-1.5 py-0.5 rounded bg-white/[0.06] border border-white/[0.1] text-[10px] font-mono text-zinc-300">
                ↵
              </kbd>
              <span className="text-zinc-400 ml-1">Select</span>
            </span>
          </div>
          <span className="text-zinc-400 font-mono text-[10px]">Potato Gateway v1.0</span>
        </div>
      </div>
    </div>
  )
}
