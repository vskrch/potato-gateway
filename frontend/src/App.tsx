import React, { useState, useCallback, useEffect, Suspense } from 'react'
import Sidebar from './components/Sidebar'
import AuthModal, { type AuthSession } from './components/AuthModal'
import ErrorBoundary from './components/ErrorBoundary'
import { Toast, Button, OfflineBanner, Toaster, QuickCopyPill } from './components/ui'
import CommandPalette from './components/CommandPalette'
import { useAuth, useSSE } from './hooks/useApi'
import { useToastQueue } from './hooks/useToast'
import { api, ap } from './lib/api'
import { RefreshCw, Radio, ShieldAlert, Menu, Search } from 'lucide-react'

import DashboardPage from './pages/DashboardPage'
import AnalyticsOverviewPage from './pages/AnalyticsOverviewPage'
import RequestsPage from './pages/RequestsPage'
import LiveFeedPage from './pages/LiveFeedPage'
import IntentsPage from './pages/IntentsPage'
import CostPage from './pages/CostPage'
import ProvidersPage from './pages/ProvidersPage'
import HealthPage from './pages/HealthPage'
import ModelsPage from './pages/ModelsPage'
import RoutingPage from './pages/RoutingPage'
import ModelLaddersPage from './pages/ModelLaddersPage'
import ModelPoolGatingPage from './pages/ModelPoolGatingPage'
import RLPage from './pages/RLPage'
import PlaygroundPage from './pages/PlaygroundPage'
import UsersPage from './pages/UsersPage'
import AccountPage from './pages/AccountPage'
import ChatPage from './pages/ChatPage'

const PAGE_META: Record<string, { title: string; subtitle: string }> = {
  chat: { title: 'Chat', subtitle: 'Claude-style chat with auto-router models' },
  dashboard: { title: 'System Overview', subtitle: 'Real-time gateway metrics & execution status' },
  analytics: { title: 'Analytics Center', subtitle: 'Throughput, latency, and operational telemetry' },
  requests: { title: 'Request Explorer', subtitle: 'Detailed request traces & execution payloads' },
  live: { title: 'Live Event Stream', subtitle: 'Real-time SSE event pipeline feed' },
  intents: { title: 'Intent Intelligence', subtitle: 'Intent classification breakdown & performance' },
  cost: { title: 'Cost & Savings', subtitle: 'Token expenditure & auto-router financial optimization' },
  playground: { title: 'API Playground', subtitle: 'Test model routing, streaming & capabilities' },
  account: { title: 'API Keys & Account', subtitle: 'Manage API tokens and access credentials' },
  users: { title: 'User Management', subtitle: 'Control platform permissions & approvals' },
  providers: { title: 'LLM Providers', subtitle: 'Manage provider API credentials & concurrency' },
  health: { title: 'Provider Health', subtitle: 'Latency telemetry, cooldowns & circuit breakers' },
  models: { title: 'Model Catalog', subtitle: 'Live model pool, quality scores & ELO rankings' },
  routing: { title: 'Routing Strategy', subtitle: 'Custom preference chains & ladder configuration' },
  ladders: { title: 'Model Ladders', subtitle: 'Drag-and-drop custom fallback chains per virtual model' },
  pools: { title: 'Model Pool Gating', subtitle: 'Per-model intent gating & auto-router inclusion' },
  rl: { title: 'Adaptive RL', subtitle: 'LinUCB contextual bandit telemetry & feature weights' },
}

export default function App() {
  const {
    ready, authed, showAuth, session, applySession, logout, isAdmin, status, email,
  } = useAuth()
  const [page, setPage] = useState('dashboard')
  const [refreshing, setRefreshing] = useState(false)
  const [mobileNavOpen, setMobileNavOpen] = useState(false)
  const [cmdOpen, setCmdOpen] = useState(false)
  const sse = useSSE()
  const { toasts, show: showToast, dismiss } = useToastQueue()

  // ⌘K global shortcut for Command Palette
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault()
        setCmdOpen(open => !open)
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [])

  // ponytail: /chat path → full-screen chat (no sidebar). Dashboard nav sets page='chat'.
  useEffect(() => {
    if (window.location.pathname === '/chat') setPage('chat')
  }, [])

  // Close the mobile drawer whenever the page changes
  const handlePageChange = useCallback((p: string) => {
    setPage(p)
    setMobileNavOpen(false)
  }, [])

  async function handleRefreshAll() {
    if (!isAdmin) return
    setRefreshing(true)
    showToast('Refreshing catalog & rankings...')
    const r = await ap('/admin/catalog/refresh', {})
    setRefreshing(false)
    if (r && (r as Record<string, unknown>).ok !== false) {
      showToast('Catalog & model ladder updated successfully')
    } else {
      showToast('Catalog refresh finished with warnings', 'err')
    }
  }

  const refreshSession = useCallback(async () => {
    const me = await api<AuthSession>('/auth/me')
    if (me?.authenticated) applySession(me)
  }, [applySession])

  const pending = status === 'pending_approval' || status === 'unverified'
  const pageMeta = PAGE_META[page] || { title: page, subtitle: 'System Module' }

  if (!ready) {
    return (
      <div className="h-screen w-screen flex flex-col items-center justify-center bg-zinc-950 text-zinc-400 gap-4">
        <div className="w-10 h-10 border-2 border-violet-500/30 border-t-violet-500 rounded-full animate-spin" />
        <span className="text-sm font-medium tracking-wide">Initializing Potato Gateway…</span>
      </div>
    )
  }

  const pageContent = (
    <Suspense fallback={<div className="p-12 text-center text-zinc-500 text-xs">Loading view…</div>}>
      {page === 'dashboard' && <DashboardPage onRefresh={handleRefreshAll} />}
      {page === 'analytics' && <AnalyticsOverviewPage />}
      {page === 'requests' && <RequestsPage />}
      {page === 'live' && <LiveFeedPage />}
      {page === 'intents' && <IntentsPage />}
      {page === 'cost' && <CostPage />}
      {page === 'account' && <AccountPage session={session} onRefresh={refreshSession} />}
      {page === 'users' && isAdmin && <UsersPage />}
      {page === 'providers' && isAdmin && <ProvidersPage />}
      {page === 'health' && isAdmin && <HealthPage />}
      {page === 'models' && isAdmin && <ModelsPage />}
      {page === 'routing' && isAdmin && <RoutingPage />}
      {page === 'ladders' && isAdmin && <ModelLaddersPage />}
      {page === 'pools' && isAdmin && <ModelPoolGatingPage />}
      {page === 'rl' && isAdmin && <RLPage />}
      {page === 'playground' && <PlaygroundPage />}
    </Suspense>
  )

  // Full-screen chat mode — no sidebar, no header. Claude-style standalone app.
  if (page === 'chat') {
    return (
      <>
        <OfflineBanner />
        <ChatPage />
        {showAuth && !authed && <AuthModal onSession={applySession} />}
        {toasts.map(t => (
          <Toast key={t.id} message={t.message} type={t.type} onDismiss={() => dismiss(t.id)} duration={t.duration} />
        ))}
      </>
    )
  }

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-zinc-950 text-zinc-100 font-sans antialiased">
      {/* Sidebar */}
      <Sidebar
        page={page}
        onNavigate={handlePageChange}
        isAdmin={isAdmin}
        email={email}
        onLogout={logout}
        mobileOpen={mobileNavOpen}
        onMobileClose={() => setMobileNavOpen(false)}
        onOpenCommandPalette={() => setCmdOpen(true)}
        liveModelCount={sse?.live_models}
      />

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col min-w-0 bg-zinc-950 relative overflow-hidden">
        {/* Top Navigation Header */}
        <header className="h-16 border-b border-white/[0.08] flex items-center justify-between px-4 sm:px-6 lg:px-8 bg-zinc-950/80 backdrop-blur-xl z-20 shrink-0 gap-3">
          <div className="flex items-center gap-2 min-w-0">
            {/* Mobile nav toggle */}
            <button
              type="button"
              onClick={() => setMobileNavOpen(true)}
              className="lg:hidden p-2 -ml-1 rounded-lg text-zinc-400 hover:text-white hover:bg-white/[0.06] transition-colors shrink-0"
              aria-label="Open navigation menu"
            >
              <Menu className="w-5 h-5" />
            </button>
            <div className="min-w-0">
              <h2 className="text-sm sm:text-base font-bold text-white tracking-tight truncate">{pageMeta.title}</h2>
              <p className="text-[11px] text-zinc-400 font-medium truncate hidden sm:block">{pageMeta.subtitle}</p>
            </div>
          </div>

          {/* Center Quick Search & Jump */}
          <div className="hidden md:flex items-center gap-2.5">
            <button
              type="button"
              onClick={() => setCmdOpen(true)}
              className="flex items-center gap-2 px-3 py-1.5 rounded-xl bg-zinc-900/90 hover:bg-zinc-800/90 border border-white/[0.08] hover:border-violet-500/40 text-xs text-zinc-400 hover:text-zinc-200 transition-all shadow-inner group"
            >
              <Search className="w-3.5 h-3.5 text-zinc-500 group-hover:text-violet-400 transition-colors" />
              <span className="font-medium">Quick search or jump...</span>
              <kbd className="ml-2 font-mono text-[10px] bg-white/[0.06] border border-white/[0.1] px-1.5 py-0.5 rounded text-zinc-300">⌘K</kbd>
            </button>
            <QuickCopyPill text={`${window.location.origin}/v1`} label="Base URL" />
          </div>

          <div className="flex items-center gap-2 sm:gap-4 shrink-0">
            {/* Live Connection Status Badge */}
            <div className="flex items-center gap-2 px-2.5 sm:px-3 py-1.5 rounded-full bg-zinc-900 border border-white/[0.08] text-xs shadow-inner">
              <span className={`w-2 h-2 rounded-full shrink-0 ${sse ? 'bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.8)] animate-pulse' : 'bg-zinc-500'}`} />
              <span className="text-zinc-300 font-mono text-[11px] hidden sm:inline">
                {sse ? (
                  <span className="flex items-center gap-1">
                    <Radio className="w-3 h-3 text-emerald-400" />
                    <strong className="text-white">{sse.live_models}</strong> models · <strong className="text-white">{sse.active_providers}</strong> providers
                  </span>
                ) : authed ? 'Connecting SSE…' : 'Disconnected'}
              </span>
              <span className="text-zinc-300 font-mono text-[11px] sm:hidden">
                {sse ? <strong className="text-white">{sse.live_models}</strong> : '—'}
              </span>
            </div>

            {/* Quick Admin Actions */}
            {isAdmin && (
              <Button
                variant="default"
                size="sm"
                onClick={handleRefreshAll}
                disabled={refreshing}
                title="Trigger catalog & ladder re-ranking"
                className="hidden sm:inline-flex"
              >
                <RefreshCw className={`w-3.5 h-3.5 ${refreshing ? 'animate-spin text-violet-400' : ''}`} />
                <span>Refresh Catalog</span>
              </Button>
            )}
            {isAdmin && (
              <button
                type="button"
                onClick={handleRefreshAll}
                disabled={refreshing}
                className="sm:hidden p-2 rounded-lg bg-zinc-800/80 border border-white/[0.1] text-zinc-200 active:scale-[0.98] disabled:opacity-50"
                aria-label="Refresh catalog"
              >
                <RefreshCw className={`w-4 h-4 ${refreshing ? 'animate-spin text-violet-400' : ''}`} />
              </button>
            )}
          </div>
        </header>

        {/* Warning Banner for Unapproved / Unverified Accounts */}
        {pending && (
          <div className="mx-4 sm:mx-6 lg:mx-8 mt-4 rounded-2xl border border-amber-500/30 bg-amber-500/10 p-4 text-xs text-amber-200 flex items-start gap-3">
            <ShieldAlert className="w-5 h-5 text-amber-400 shrink-0 mt-0.5" />
            <div>
              <strong className="font-semibold text-amber-100">Account Action Needed: </strong>
              {status === 'unverified'
                ? 'Please check your email to verify your address. Once verified, an administrator will approve your API access.'
                : 'Your account is currently pending administrator approval. API keys will be issued once approved.'}
            </div>
          </div>
        )}

        {/* Dynamic Page Container */}
        <ErrorBoundary onReset={() => window.location.reload()}>
          <main className="flex-1 overflow-y-auto p-4 sm:p-6 lg:p-8 custom-scrollbar">
            {pageContent}
          </main>
        </ErrorBoundary>
      </div>

      <OfflineBanner />

      {/* Modals & Toasts */}
      {showAuth && !authed && <AuthModal onSession={applySession} />}

      {/* ⌘K Command Palette */}
      <CommandPalette
        open={cmdOpen}
        onClose={() => setCmdOpen(false)}
        onNavigate={handlePageChange}
        isAdmin={isAdmin}
        onRefreshCatalog={isAdmin ? handleRefreshAll : undefined}
      />

      {toasts.map(t => (
        <Toast key={t.id} message={t.message} type={t.type} onDismiss={() => dismiss(t.id)} duration={t.duration} />
      ))}

      {/* shadcn Sonner toaster — available app-wide via toast.success/error */}
      <Toaster />
    </div>
  )
}
