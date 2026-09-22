import React from 'react'
import { clsx } from 'clsx'
import {
  LayoutDashboard,
  BarChart3,
  ListFilter,
  Radio,
  BrainCircuit,
  Coins,
  Terminal,
  MessageSquare,
  User,
  Users,
  Server,
  Activity,
  Cpu,
  GitFork,
  LogOut,
  Zap,
  ShieldCheck,
  Key,
  Layers,
  Filter,
  Search
} from 'lucide-react'

interface SidebarProps {
  page: string
  onNavigate: (page: string) => void
  isAdmin?: boolean
  email?: string | null
  onLogout?: () => void
  onOpenCommandPalette?: () => void
  liveModelCount?: number
  // Mobile drawer control
  mobileOpen?: boolean
  onMobileClose?: () => void
}

interface NavItem {
  id: string
  label: string
  icon: React.ComponentType<{ className?: string }>
  badge?: string | number
  pulse?: boolean
}

const ANALYTICS_NAV: NavItem[] = [
  { id: 'dashboard', label: 'Overview', icon: LayoutDashboard },
  { id: 'analytics', label: 'Analytics', icon: BarChart3 },
  { id: 'requests', label: 'Requests', icon: ListFilter },
  { id: 'live', label: 'Live Stream', icon: Radio, pulse: true },
  { id: 'intents', label: 'Intents', icon: BrainCircuit },
  { id: 'cost', label: 'Cost Center', icon: Coins },
]

const DEV_NAV: NavItem[] = [
  { id: 'chat', label: 'Chat', icon: MessageSquare },
  { id: 'playground', label: 'Playground', icon: Terminal },
  { id: 'account', label: 'API Keys & Account', icon: Key },
]

const ADMIN_NAV: NavItem[] = [
  { id: 'users', label: 'Users', icon: Users },
  { id: 'providers', label: 'LLM Providers', icon: Server },
  { id: 'health', label: 'Provider Health', icon: Activity },
  { id: 'models', label: 'Model Catalog', icon: Cpu },
  { id: 'routing', label: 'Routing Engine', icon: GitFork },
  { id: 'ladders', label: 'Model Ladders', icon: Layers },
  { id: 'pools', label: 'Model Pool Gating', icon: Filter },
  { id: 'rl', label: 'Adaptive RL', icon: Zap },
]

export default function Sidebar({
  page,
  onNavigate,
  isAdmin,
  email,
  onLogout,
  onOpenCommandPalette,
  liveModelCount,
  mobileOpen,
  onMobileClose,
}: SidebarProps) {
  const renderNavGroup = (title: string, items: NavItem[]) => (
    <div className="mb-5">
      <div className="px-3 mb-1.5 text-[10px] font-semibold uppercase tracking-wider text-zinc-400">
        {title}
      </div>
      <div className="space-y-0.5">
        {items.map(item => {
          const Icon = item.icon
          const isActive = page === item.id
          const dynamicBadge = item.id === 'models' && liveModelCount ? liveModelCount : item.badge

          return (
            <button
              key={item.id}
              onClick={() => {
                onNavigate(item.id)
                onMobileClose?.()
              }}
              aria-current={isActive ? 'page' : undefined}
              className={clsx(
                'w-full flex items-center justify-between px-3 py-2 rounded-xl text-[13px] font-medium transition-all duration-200 text-left group relative',
                isActive
                  ? 'bg-gradient-to-r from-violet-500/20 via-violet-500/10 to-transparent text-white border border-violet-500/30 shadow-[0_0_20px_rgba(139,92,246,0.18)] font-semibold'
                  : 'text-zinc-400 hover:bg-white/[0.04] hover:text-zinc-200 border border-transparent'
              )}
            >
              {/* Active glow indicator on the left */}
              {isActive && (
                <span className="absolute left-0 top-1/2 -translate-y-1/2 w-1 h-5 bg-gradient-to-b from-violet-400 to-fuchsia-500 rounded-r-full shadow-[0_0_10px_#8b5cf6]" />
              )}
              <div className="flex items-center gap-2.5 min-w-0">
                <Icon
                  className={clsx(
                    'w-4 h-4 transition-colors shrink-0',
                    isActive ? 'text-violet-400' : 'text-zinc-500 group-hover:text-zinc-300'
                  )}
                />
                <span className="truncate">{item.label}</span>
              </div>

              {/* Dynamic badge or pulse */}
              <div className="flex items-center gap-1.5 shrink-0">
                {item.pulse && (
                  <span className="relative flex h-2 w-2">
                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
                    <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500 shadow-[0_0_6px_rgba(16,185,129,0.8)]" />
                  </span>
                )}
                {dynamicBadge !== undefined && (
                  <span
                    className={clsx(
                      'text-[10px] font-mono px-1.5 py-0.2 rounded font-semibold',
                      isActive
                        ? 'bg-violet-500/30 text-violet-200 border border-violet-500/40'
                        : 'bg-white/[0.06] text-zinc-400 group-hover:text-zinc-300'
                    )}
                  >
                    {dynamicBadge}
                  </span>
                )}
              </div>
            </button>
          )
        })}
      </div>
    </div>
  )

  return (
    <>
      {/* Mobile backdrop */}
      {mobileOpen && (
        <div
          className="fixed inset-0 bg-black/70 backdrop-blur-md z-30 lg:hidden"
          onClick={onMobileClose}
          aria-hidden="true"
        />
      )}
      <aside
        className={clsx(
          'bg-zinc-950/95 border-r border-white/[0.08] flex flex-col z-40 select-none shadow-[4px_0_24px_rgba(0,0,0,0.4)]',
          // Desktop: fixed-width column
          'w-64 shrink-0',
          // Mobile: slide-in drawer
          'fixed inset-y-0 left-0 transition-transform duration-300 ease-out',
          mobileOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0 lg:static lg:translate-x-0'
        )}
        aria-label="Primary navigation"
      >
        {/* Brand Header */}
        <div className="px-5 py-4 flex items-center gap-3 border-b border-white/[0.06] bg-white/[0.01]">
          <div className="w-10 h-10 bg-gradient-to-br from-violet-600 via-fuchsia-600 to-indigo-700 rounded-xl flex items-center justify-center shadow-[0_0_25px_rgba(139,92,246,0.6)] border border-white/20 shrink-0">
            <Zap className="w-5 h-5 text-white fill-white" />
          </div>
          <div className="min-w-0">
            <div className="flex items-center gap-1.5">
              <h1 className="text-sm font-bold tracking-tight text-white truncate">Potato Gateway</h1>
              <span className="text-[9px] font-mono font-bold px-1.5 py-0.2 bg-gradient-to-r from-violet-500/20 to-fuchsia-500/20 text-violet-300 rounded border border-violet-500/30 shrink-0">
                HA
              </span>
            </div>
            <p className="text-[11px] text-zinc-400 font-medium flex items-center gap-1 mt-0.5">
              {isAdmin ? (
                <span className="text-emerald-400 flex items-center gap-1">
                  <ShieldCheck className="w-3 h-3" /> Enterprise Admin
                </span>
              ) : (
                <span className="text-zinc-400">99.99% Resilient Proxy</span>
              )}
            </p>
          </div>
        </div>

        {/* Quick ⌘K Command Palette Trigger */}
        {onOpenCommandPalette && (
          <div className="px-3 pt-3 pb-1">
            <button
              onClick={onOpenCommandPalette}
              className="w-full flex items-center justify-between gap-2 px-3 py-2 rounded-xl bg-white/[0.03] hover:bg-white/[0.07] border border-white/[0.07] hover:border-violet-500/30 text-xs text-zinc-400 hover:text-zinc-200 transition-all duration-150 group shadow-inner"
            >
              <div className="flex items-center gap-2">
                <Search className="w-3.5 h-3.5 text-zinc-500 group-hover:text-violet-400 transition-colors" />
                <span className="text-[12px] font-medium">Search or jump...</span>
              </div>
              <kbd className="text-[10px] font-mono font-semibold px-1.5 py-0.5 rounded bg-white/[0.06] border border-white/[0.08] text-zinc-400 group-hover:text-zinc-300">
                ⌘K
              </kbd>
            </button>
          </div>
        )}

        {/* Navigation Groups */}
        <nav className="flex-1 px-3 py-3 overflow-y-auto custom-scrollbar space-y-1">
          {renderNavGroup('Analytics', ANALYTICS_NAV)}
          {renderNavGroup('Developer', DEV_NAV)}
          {isAdmin && renderNavGroup('Administration', ADMIN_NAV)}
        </nav>

        {/* User Footer */}
        <div className="p-3 border-t border-white/[0.08] bg-zinc-950/60">
          <div className="flex items-center justify-between gap-2 p-2 rounded-xl bg-zinc-900/70 border border-white/[0.06] hover:border-white/[0.12] transition-colors">
            <div className="flex items-center gap-2.5 min-w-0">
              <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-violet-500/30 to-fuchsia-500/20 border border-violet-500/40 flex items-center justify-center text-violet-200 text-xs font-semibold shrink-0">
                {email ? email.charAt(0).toUpperCase() : <User className="w-3.5 h-3.5" />}
              </div>
              <div className="min-w-0">
                <p className="text-xs font-medium text-zinc-200 truncate">{email || 'Authenticated User'}</p>
                <div className="flex items-center gap-1">
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
                  <p className="text-[10px] text-zinc-400 capitalize">{isAdmin ? 'Administrator' : 'Standard User'}</p>
                </div>
              </div>
            </div>
            {onLogout && (
              <button
                type="button"
                onClick={onLogout}
                className="p-1.5 text-zinc-400 hover:text-rose-400 hover:bg-rose-500/10 rounded-lg transition-colors shrink-0"
                title="Sign Out"
                aria-label="Sign Out"
              >
                <LogOut className="w-4 h-4" />
              </button>
            )}
          </div>
        </div>
      </aside>
    </>
  )
}
